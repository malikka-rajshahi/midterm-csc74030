"""Fit global training-sized windows and evaluate class-independent ROI containment."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from .generate_rois import generate_proposals, load_config


def read_training(data):
    table=pd.read_parquet(Path(data)/'annotations.parquet')
    table=table[table.image_path.str.startswith('train/')].copy()
    coords=np.stack(table.bbox_coords).astype(float)
    if not np.isfinite(coords).all() or (coords[:,2:]<=0).any():
        raise ValueError('Invalid training boxes.')
    table[['x','y','w','h']]=coords
    return table.drop_duplicates(['image_path','label','x','y','w','h'])


def select_images(table,limit):
    paths=sorted(table.image_path.unique())
    if limit<=0 or limit>=len(paths):
        return paths
    groups={}
    rng=np.random.default_rng(42)
    for path in paths:
        groups.setdefault(path.split('/')[1],[]).append(path)
    for group in groups.values():
        rng.shuffle(group)
    selected=[]
    while len(selected)<limit:
        for camera in sorted(groups):
            if groups[camera] and len(selected)<limit:
                selected.append(groups[camera].pop())
    return selected


def fit_config(data,output,work_size=(1440,930)):
    """Global shape priors from canonical official-train boxes, never test labels."""
    data=Path(data)
    table=read_training(data)
    ww,wh=work_size
    shapes={}
    for path in sorted(table.image_path.unique()):
        with Image.open(data/path) as image:
            shapes[path]=image.size
    original=np.array([shapes[path] for path in table.image_path])
    xy=table[['x','y']].to_numpy()
    sizes=table[['w','h']].to_numpy()
    if (xy<0).any() or (xy+sizes>original).any():
        raise ValueError('Out-of-image GT.')
    scaled=sizes*np.array([ww,wh])/original
    table[['work_w','work_h']]=scaled
    bank=set()
    statistics=[]
    for label,group in table.groupby('label'):
        values=group[['work_w','work_h']].to_numpy()
        area=values.prod(axis=1)
        cuts=np.quantile(area,[0,.1,.25,.5,.75,.9,1])
        for index,(low,high) in enumerate(zip(cuts[:-1],cuts[1:])):
            subset=values[(area>=low)&((area<=high) if index==5 else (area<high))]
            if len(subset):
                # Context allowance is global, not fitted to individual evaluation GT.
                w,h=np.maximum(4,np.ceil(np.median(subset,axis=0)*1.25)).astype(int)
                bank.add((min(ww,int(w)),min(wh,int(h))))
        statistics.append(dict(source_label=label,boxes=len(group),
            width_p10_p50_p90=np.percentile(values[:,0],[10,50,90]).round(2).tolist(),
            height_p10_p50_p90=np.percentile(values[:,1],[10,50,90]).round(2).tolist()))
    config=dict(version=2,work_size=list(work_size),window_sizes=sorted(bank,key=lambda wh:wh[0]*wh[1]),
        saturation_min=.35,value_min=.15,anchor_blocks=[2,4,8],min_anchor_pixels=4,
        anchor_padding=.25,nearest_windows=3,max_roi_area_fraction=.25,
        merge_overlap=.5,merge_growth=1.5,merge_scale_ratio=4.,window_anchor_area_ratio=4.,
        training_provenance=dict(images=len(shapes),canonical_boxes=len(table),
            annotation_sha256=hashlib.sha256((data/'annotations.parquet').read_bytes()).hexdigest(),
            fit='Per-label area-quantile bins; median width/height times 1.25. Labels used only for global bank balancing.',
            gt_statistics=statistics))
    Path(output).write_text(json.dumps(config,indent=2))
    return config


def measure_gt(gt,proposals):
    """Independent containment; compactness is conditional on satisfying coverage."""
    gt=np.asarray(gt,float)
    proposals=np.asarray(proposals,float).reshape(-1,4)
    gt_area=np.prod(gt[2:]-gt[:2])
    area=np.prod(proposals[:,2:]-proposals[:,:2],axis=1)
    intersection=np.maximum(0,np.minimum(gt[2:],proposals[:,2:])-np.maximum(gt[:2],proposals[:,:2])).prod(axis=1)
    coverage=intersection/gt_area
    contains=(proposals[:,:2]<=gt[:2]).all(axis=1)&(proposals[:,2:]>=gt[2:]).all(axis=1)
    iou=intersection/np.maximum(gt_area+area-intersection,1e-30)
    result=dict(best_gt_coverage=float(coverage.max(initial=0)),covered_95=bool((coverage>=.95).any()),
                covered_100=bool(contains.any()),best_iou=float(iou.max(initial=0)))
    for suffix,valid in [('95',coverage>=.95),('100',contains)]:
        eligible=np.flatnonzero(valid)
        index=int(eligible[np.argmin(area[eligible])]) if len(eligible) else -1
        result[f'compact_roi_{suffix}']=index
        result[f'area_ratio_{suffix}']=float(area[index]/gt_area) if index>=0 else np.nan
    return result


def draw_boxes(ax,boxes,color):
    boxes=np.asarray(boxes).reshape(-1,4)
    if not len(boxes):
        return
    a,b,c,d=boxes.T
    vertices=np.stack([np.c_[a,b],np.c_[c,b],np.c_[c,d],np.c_[a,d],np.c_[a,b]],axis=1)
    ax.add_collection(LineCollection(vertices,colors=color,linewidths=.7))


def plot_image(image,truth,variants,records,path,title):
    fig,axes=plt.subplots(1,len(variants)+1,figsize=(5*(len(variants)+1),6),layout='constrained')
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    for ax in axes:
        ax.imshow(rgb)
        ax.axis('off')
    draw_boxes(axes[0],truth,'cyan')
    axes[0].set_title(f'Source GT boxes: {len(truth)} (unverified)')
    for ax,(name,result) in zip(axes[1:],variants.items()):
        draw_boxes(ax,result['work_boxes'],'lime')
        scores=records[name]
        missed=np.array([not row['covered_95'] for row in scores])
        draw_boxes(ax,truth[missed],'orange')
        axes_title=f'{name}\nROIs: {len(result["work_boxes"])}; GT >=95%: {sum(r["covered_95"] for r in scores)}/{len(truth)}'
        ax.set_title(axes_title)
    fig.suptitle(title+'\nCyan: supplied GT | Green: all candidate ROIs | Orange: GT not covered >=95%. No classification.')
    fig.savefig(path,dpi=110)
    plt.close(fig)


def evaluate(data,output,config=None,limit=12,apply_denoise=True):
    data,output=Path(data),Path(output)
    cfg=load_config(config)
    table=read_training(data)
    selected=select_images(table,limit)
    (output/'images').mkdir(parents=True,exist_ok=True)
    rows,counts,gallery=[],[],[]
    settings={'no_background_filter':dict(background_mode='none'),
              'green_40':dict(background_mode='green_40'),
              'green_blue_50_unmerged':dict(background_mode='green_blue_50',merge=False),
              'green_blue_50':dict(background_mode='green_blue_50'),
              'green_blue_50_s25':dict(background_mode='green_blue_50',saturation_min=.25)}
    roi_file=(output/'proposals.jsonl').open('w')
    try:
        for number,path in enumerate(selected):
            image=cv2.imread(str(data/path))
            if image is None:
                raise ValueError(f'Cannot read training image: {path}')
            group=table[table.image_path==path]
            height,width=image.shape[:2]
            gt=group[['x','y','w','h']].to_numpy().copy()
            gt[:,2:]+=gt[:,:2]
            if (gt<0).any() or (gt[:,[0,2]]>width).any() or (gt[:,[1,3]]>height).any():
                raise ValueError('Out-of-image GT.')
            ww,wh=cfg['work_size']
            gt*=np.array([ww/width,wh/height,ww/width,wh/height])
            variants,records={},{}
            for method,options in settings.items():
                options=options.copy()
                run_config=dict(cfg,saturation_min=options.pop('saturation_min',cfg['saturation_min']))
                result=generate_proposals(image,config=run_config,apply_denoise=apply_denoise,return_masks=True,**options)
                variants[method]=result
                records[method]=[]
                counts.append(dict(image=path,method=method,**result['counts'],runtime_seconds=result['runtime_seconds']))
                roi_file.write(json.dumps(dict(image=path,method=method,work_boxes=result['work_boxes'].tolist(),
                                              original_boxes=result['boxes'].tolist()))+'\n')
                for index,(label,box) in enumerate(zip(group.label,gt)):
                    record=measure_gt(box,result['work_boxes'])
                    x0,y0=np.floor(box[:2]).astype(int)
                    x1,y1=np.ceil(box[2:]).astype(int)
                    support_pixels=int(result['color_masks']['support'][y0:y1,x0:x1].sum())
                    record['red_orange_pixels_in_gt']=support_pixels
                    record['diagnosis']='covered_95' if record['covered_95'] else 'little_red_orange_in_gt' if support_pixels<cfg['min_anchor_pixels'] else 'not_contained_despite_red_orange'
                    records[method].append(record)
                    rows.append(dict(image=path,object_index=index,source_label=label,method=method,
                                     gt_work_xyxy=json.dumps(box.tolist()),
                                     gt_area_fraction=float(np.prod(box[2:]-box[:2])/(ww*wh)),**record))
            filename=f'{number:03d}_{Path(path).stem}.png'
            plot_image(variants['green_blue_50']['work_image_bgr'],gt,variants,records,output/'images'/filename,path)
            gallery.append((path,filename))
            print(f'{number+1}/{len(selected)} training images; default V2 ROIs {variants["green_blue_50"]["counts"]["final"]}',flush=True)
    finally:
        roi_file.close()
    objects=pd.DataFrame(rows)
    per_image=pd.DataFrame(counts)
    objects.to_csv(output/'per_gt.csv',index=False)
    per_image.to_csv(output/'per_image.csv',index=False)
    objects.groupby(['method','diagnosis']).size().rename('gt_boxes').to_csv(output/'diagnosis.csv')
    summaries=[]
    by_label=[]
    def summarize(group):
        return dict(gt_boxes=len(group),
            compact_coverage_95_area_le_4=float((group.covered_95 & (group.area_ratio_95<=4)).mean()),gt_coverage_95=float(group.covered_95.mean()),
            gt_coverage_100=float(group.covered_100.mean()),iou_05=float((group.best_iou>=.5).mean()),
            area_ratio_95_median=float(group.area_ratio_95.median()),area_ratio_95_p90=float(group.area_ratio_95.quantile(.9)),
            area_ratio_100_median=float(group.area_ratio_100.median()))
    for method,group in objects.groupby('method',sort=False):
        count=per_image[per_image.method==method]
        summaries.append(dict(method=method,**summarize(group),mean_roi_count=float(count.final.mean()),
                              mean_runtime_seconds=float(count.runtime_seconds.mean())))
    for (method,label),group in objects.groupby(['method','source_label']):
        by_label.append(dict(method=method,source_label=label,**summarize(group)))
    summary=pd.DataFrame(summaries)
    summary.to_csv(output/'comparison.csv',index=False)
    pd.DataFrame(by_label).to_csv(output/'by_label.csv',index=False)
    for metric in ['covered_95','covered_100']:
        pivot=objects.pivot(index=['image','object_index'],columns='method',values=metric)
        for method in summary.method:
            summary.loc[summary.method==method,f'lost_vs_no_filter_{metric}']=int((pivot.no_background_filter&~pivot[method]).sum())
    summary.to_csv(output/'comparison.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(18,5),layout='constrained')
    labels=summary.method.str.replace('_',' ')
    x=np.arange(len(summary))
    axes[0].bar(x,summary.mean_roi_count)
    axes[0].set_title('Mean candidate count')
    axes[1].bar(x-.18,summary.gt_coverage_95*100,width=.36,label='GT >=95%')
    axes[1].bar(x+.18,summary.gt_coverage_100*100,width=.36,label='GT 100%')
    axes[1].set_ylim(0,100)
    axes[1].legend()
    axes[1].set_title('GT coverage (%)')
    axes[2].bar(x,summary.area_ratio_95_median)
    axes[2].set_title('Median compact ROI / GT area (>=95%)')
    for ax in axes:
        ax.set_xticks(x,labels,rotation=20,ha='right')
    fig.suptitle('ROI V2: training development sample, no classification')
    fig.savefig(output/'summary.png',dpi=140)
    plt.close(fig)
    report=dict(version=2,scope='official training development sample only; test images never opened',
        images=len(selected),canonical_gt_boxes=len(objects)//len(settings),denoised=apply_denoise,
        config=cfg,variants=settings,evaluation='GT intersection / GT area; strict containment uses coordinate inequalities.',
        timing='Per variant end-to-end resize, denoise, color, windows, filtering, merging and evidence; excludes file I/O and plotting.',
        limitations='Configuration fitted on official training GT; this is in-sample development, not a held-out accuracy estimate.')
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    default=summary[summary.method=='green_blue_50'].iloc[0]
    unmerged=summary[summary.method=='green_blue_50_unmerged'].iloc[0]
    lower_sat=summary[summary.method=='green_blue_50_s25'].iloc[0]
    rejected=int(per_image[per_image.method=='green_blue_50'].background_rejected.sum())
    analysis=(f"Default: {default.gt_coverage_95:.1%} GT coverage >=95%, "
              f"{default.gt_coverage_100:.1%} strict containment, {default.mean_roi_count:.1f} ROIs/image. "
              f"Median/p90 compact area ratio at >=95%: {default.area_ratio_95_median:.2f}/{default.area_ratio_95_p90:.2f}. "
              f"Merging reduces {unmerged.mean_roi_count:.1f} unmerged ROIs/image but makes compact ROIs larger. "
              f"The green/blue gate rejects {rejected} pre-merge candidates in this run; do not assume a filtering benefit. "
              f"Lowering S to 0.25 yields {lower_sat.gt_coverage_95:.1%} coverage with area-ratio p90 "
              f"{lower_sat.area_ratio_95_p90:.2f}: greater coverage can come from oversized boxes. "
              "These settings are provisional; the small, imbalanced training sample cannot establish an optimum.")
    (output/'analysis.md').write_text('# ROI V2 development diagnosis\n\n'+analysis+'\n')
    body=['<!doctype html><html lang="en"><meta charset="utf-8"><title>ROI V2 containment</title><style>body{font:16px system-ui;margin:24px}img{width:100%}table{border-collapse:collapse}td,th{padding:7px}</style>',
        '<h1>ROI V2: candidate containment</h1><p>Training development sample only; supplied GT is unverified and can include background/groups. No classification, intensity or precision/F1. GT is never supplied to image-specific proposal generation.</p>',
        '<p>Green-blue-50-s25 changes only the saturation lower bound from 0.35 to 0.25; background-50 comparisons use 0.35. Diagnosis categories describe observed support, not verified failure causes.</p>',
        '<p>Green and blue refer to HSV pixel membership, not semantic background masks. Merging is bounded against a fixed original seed and rechecks background. Compact area ratios use the smallest eligible ROI for each covered GT; uncovered GT has no ratio.</p>',
        '<p>'+html.escape(analysis)+'</p>',
        summary.to_html(index=False,float_format=lambda v:f'{v:.3f}'),'<img src="summary.png">']
    body += [f'<h3>{html.escape(path)}</h3><img loading="lazy" src="images/{filename}">' for path,filename in gallery]
    (output/'index.html').write_text('\n'.join(body)+'</html>')
    print(summary.to_string(index=False),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path.home()/'Downloads/datasets/Roadwork_Cones_Dataset/roadwork_cones_dataset')
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('roi_config.json'))
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('results'))
    parser.add_argument('--limit',type=int,default=12,help='0 means all official training images.')
    parser.add_argument('--fit-config',action='store_true',help='Refit global window bank from official training GT before evaluation.')
    parser.add_argument('--no-denoise',action='store_true',help='Explicit preprocessing ablation.')
    args=parser.parse_args()
    if args.fit_config:
        fit_config(args.data,args.config)
    evaluate(args.data,args.output,args.config,args.limit,not args.no_denoise)
