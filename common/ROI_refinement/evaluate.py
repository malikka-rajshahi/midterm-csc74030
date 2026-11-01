"""Training-only S1/S2/S3 geometry study inside frozen V4 official coarse ROIs."""
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
from common.ROI.v4.evaluate_rois import read_training, measure_gt, draw_boxes
from .generate_windows import generate_windows, load_config, combined_shapes


def fit_config(data, output):
    """Global size/aspect and perspective priors from official training only."""
    data=Path(data)
    table=read_training(data)
    sizes={}
    for path in sorted(table.image_path.unique()):
        with Image.open(data/path) as image:
            sizes[path]=image.size
    original=np.array([sizes[path] for path in table.image_path])
    coords=table[['x','y','w','h']].to_numpy()
    if (coords[:,:2]<0).any() or (coords[:,:2]+coords[:,2:]>original).any():
        raise ValueError('Invalid training GT.')
    work_size=[1440,930]
    wh=coords[:,2:]*work_size/original
    labels=table.label.to_numpy()
    bounds=[]
    for label in sorted(set(labels)):
        values=wh[labels==label]
        ratios=values[:,0]/values[:,1]
        bounds.append(dict(source_label=label,
            width=[float(np.quantile(values[:,0],.02)*.8),float(np.quantile(values[:,0],.98)*1.25)],
            height=[float(np.quantile(values[:,1],.02)*.8),float(np.quantile(values[:,1],.98)*1.25)],
            aspect=[float(np.quantile(ratios,.02)*.8),float(np.quantile(ratios,.98)*1.25)]))
    generic=sorted({(w,round(w*r)) for w in [4,8,16,32,64,128,256,512] for r in [.5,1,2,4]
                    if w<=work_size[0] and round(w*r)<=work_size[1]})
    def plausible(shape):
        w,h=shape
        return any(b['width'][0]<=w<=b['width'][1] and b['height'][0]<=h<=b['height'][1] and
                   b['aspect'][0]<=w/h<=b['aspect'][1] for b in bounds)
    constrained=[shape for shape in generic if plausible(shape)]
    bottom=(coords[:,1]+coords[:,3])/original[:,1]
    center=(coords[:,0]+coords[:,2]/2)/original[:,0]
    slope=max(0.,float(np.polyfit(bottom,np.log(wh[:,1]),1)[0]))
    intercept=float(np.median(np.log(wh[:,1])-slope*bottom))
    residual=np.log(wh[:,1])-intercept-slope*bottom
    low=bottom<=np.quantile(bottom,.3)
    high=bottom>=np.quantile(bottom,.7)
    perspective=dict(top_y=float(np.quantile(bottom,.01)),
        top_left=float(np.quantile(center[low],.01)),top_right=float(np.quantile(center[low],.99)),
        bottom_left=float(np.quantile(center[high],.01)),bottom_right=float(np.quantile(center[high],.99)),
        slope=slope,intercept=intercept,residual_low=float(np.quantile(residual,.05)),
        residual_high=float(np.quantile(residual,.95)),
        y_log_height_correlation=float(np.corrcoef(bottom,np.log(wh[:,1]))[0,1]))
    config=dict(work_size=work_size,stride_fraction=.5,generic_shapes=generic,constrained_shapes=constrained,
        size_aspect_bounds=bounds,perspective=perspective,
        training_provenance=dict(images=len(sizes),canonical_boxes=len(table),
            annotation_sha256=hashlib.sha256((data/'annotations.parquet').read_bytes()).hexdigest(),
            scope='Official training only; same population used for development evaluation; not held-out validation'),
        geometry_definition='Rectangular windows; trapezoid constrains normalized center-x and bottom-y, not patch shape. Log height follows bottom-y.')
    config['selected_method']='FINAL'
    config['final_rule']='S2 plus S1 upper-size bounds rounded to the next power-of-two grid scale; lower/aspect bounds unchanged; no perspective gate.'
    config['final_shapes']=combined_shapes(config)
    Path(output).write_text(json.dumps(config,indent=2))
    return config


def metrics(gt, windows):
    record=measure_gt(gt,windows)
    if len(windows):
        intersection=np.maximum(0,np.minimum(gt[2:],windows[:,2:])-np.maximum(gt[:2],windows[:,:2])).prod(axis=1)
        eligible=np.flatnonzero(intersection/np.prod(gt[2:]-gt[:2])>=.95)
        if len(eligible):
            index=int(eligible[np.argmin(np.prod(windows[eligible,2:]-windows[eligible,:2],axis=1))])
        else:
            index=int(intersection.argmax()) if intersection.max()>0 else -1
    else:
        index=-1
    record['diagnostic_window_index']=index
    return record


def plot_scene(image, gt, coarse, variants, records, output, title):
    """GT-best windows are oracle diagnostics, not feature-selected predictions."""
    fig,axes=plt.subplots(1,len(variants)+1,figsize=(6*(len(variants)+1),6),layout='constrained')
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    for ax in axes:
        ax.imshow(rgb);ax.axis('off')
    draw_boxes(axes[0],coarse,'lime');draw_boxes(axes[0],gt,'cyan')
    axes[0].set_title(f'Frozen V4: {len(coarse)} coarse ROIs | GT: {len(gt)}')
    for ax,(method,result) in zip(axes[1:],variants.items()):
        windows=result['boxes'];rows=records[method]
        # Show all window centers as density; GT-best rectangles separately.
        if len(windows):
            centers=(windows[:,:2]+windows[:,2:])/2
            density,_,_=np.histogram2d(centers[:,1],centers[:,0],bins=(93,144),range=[[0,image.shape[0]],[0,image.shape[1]]])
            ax.imshow(np.ma.masked_where(density==0,np.log1p(density)),extent=(0,image.shape[1],image.shape[0],0),cmap='magma',alpha=.45)
        chosen=sorted({row['diagnostic_window_index'] for row in rows if row['diagnostic_window_index']>=0})
        draw_boxes(ax,windows[chosen] if chosen else [],'lime')
        for box,row in zip(gt,rows):
            draw_boxes(ax,box[None],'cyan' if row['covered_95'] else 'gold' if row['best_gt_coverage']>0 else 'red')
        ax.set_title(f'{method}: {len(windows):,} windows\nGT >=95%: {sum(r["covered_95"] for r in rows)}/{len(gt)}')
    fig.suptitle(title+'\nHeat: all window-center density | Green: GT-best diagnostic windows (oracle, not predictions) | Cyan/gold/red: covered/partial/no-overlap GT.')
    fig.savefig(output,dpi=120);plt.close(fig)


def evaluate(data, coarse_dir, output, config=None, methods=('FINAL',)):
    data,coarse_dir,output=Path(data),Path(coarse_dir),Path(output)
    cfg=load_config(config)
    metadata=json.loads((coarse_dir/'summary.json').read_text())
    if metadata.get('selected_method')!='V4 green_blue_50' or metadata['config']['work_size']!=cfg['work_size']:
        raise ValueError('Expected frozen official V4 green_blue_50 at the same working resolution.')
    if cfg['training_provenance']['annotation_sha256']!=metadata['config']['training_provenance']['annotation_sha256']:
        raise ValueError('Training annotations differ.')
    coarse_rows=[json.loads(line) for line in (coarse_dir/'proposals.jsonl').read_text().splitlines()]
    table=read_training(data)
    (output/'images').mkdir(parents=True,exist_ok=True)
    (output/'windows').mkdir(exist_ok=True)
    object_rows,count_rows,gallery=[],[],[]
    for number,source in enumerate(coarse_rows):
        path=source['image']
        if not path.startswith('train/') or source['method']!='green_blue_50':
            raise ValueError('Only official training coarse inputs are allowed.')
        image=cv2.imread(str(data/path))
        if image is None:raise ValueError(path)
        ih,iw=image.shape[:2];ww,wh=cfg['work_size']
        work=cv2.resize(image,(ww,wh),interpolation=cv2.INTER_AREA)
        group=table[table.image_path==path]
        gt=group[['x','y','w','h']].to_numpy().copy();gt[:,2:]+=gt[:,:2]
        gt*=np.array([ww/iw,wh/ih,ww/iw,wh/ih])
        coarse=np.array(source['work_boxes'],np.int32).reshape(-1,4)
        ceiling=[measure_gt(box,coarse) for box in gt]
        variants={method:generate_windows(coarse,cfg,method) for method in methods}
        records={}
        for method,result in variants.items():
            windows=result['boxes'];records[method]=[]
            filename=f'{number:03d}_{Path(path).stem}_{method}.npz'
            np.savez_compressed(output/'windows'/filename,work_boxes=windows)
            count_rows.append(dict(image=path,method=method,coarse_roi_count=len(coarse),windows=len(windows),
                runtime_seconds=result['runtime_seconds'],artifact='windows/'+filename))
            for index,(label,box,upper) in enumerate(zip(group.label,gt,ceiling)):
                row=metrics(box,windows);records[method].append(row)
                assert row['best_gt_coverage']<=upper['best_gt_coverage']+1e-12
                object_rows.append(dict(image=path,object_index=index,source_label=label,method=method,
                    coarse_covered_95=upper['covered_95'],coarse_covered_100=upper['covered_100'],
                    coarse_area_ratio_95=upper['area_ratio_95'],
                    gt_work_xyxy=json.dumps(box.tolist()),**row))
        filename=f'{number:03d}_{Path(path).stem}.png'
        plot_scene(work,gt,coarse,variants,records,output/'images'/filename,path)
        gallery.append((path,filename))
        print(f'{number+1}/{len(coarse_rows)}: '+', '.join(f'{m}={len(r["boxes"]):,}' for m,r in variants.items()),flush=True)
    objects=pd.DataFrame(object_rows);counts=pd.DataFrame(count_rows)
    objects.to_csv(output/'per_gt.csv',index=False);counts.to_csv(output/'per_image.csv',index=False)
    summary=[];by_label=[]
    def summarize(group):
        return dict(gt_count=len(group),covered_95=int(group.covered_95.sum()),covered_100=int(group.covered_100.sum()),
            coverage_95=float(group.covered_95.mean()),iou_05=float((group.best_iou>=.5).mean()),
            coarse_covered_95=int(group.coarse_covered_95.sum()),
            conditional_coverage_95=float(group.loc[group.coarse_covered_95,'covered_95'].mean()),
            lost_coarse_covered_95=int((group.coarse_covered_95&~group.covered_95).sum()),
            matched_coarse_ratio_95_median=float(group.loc[group.covered_95,'coarse_area_ratio_95'].median()),
            compact_ratio_95_median=float(group.area_ratio_95.median()),compact_ratio_95_p90=float(group.area_ratio_95.quantile(.9)))
    for method,group in objects.groupby('method',sort=False):
        c=counts[counts.method==method]
        summary.append(dict(method=method,**summarize(group),mean_windows=float(c.windows.mean()),
                            mean_runtime_seconds=float(c.runtime_seconds.mean())))
    for (method,label),group in objects.groupby(['method','source_label']):
        by_label.append(dict(method=method,source_label=label,**summarize(group)))
    summary=pd.DataFrame(summary)
    summary.to_csv(output/'summary.csv',index=False);pd.DataFrame(by_label).to_csv(output/'by_label.csv',index=False)
    report=dict(scope='Geometry-only in-sample development; frozen V4 input; official test untouched',config=cfg,
                coarse_source=str(coarse_dir.resolve()),coarse_predictions_sha256=hashlib.sha256((coarse_dir/'proposals.jsonl').read_bytes()).hexdigest(),
                features_tested=False,selected_default='FINAL',timing='Local window generation/filtering/exact dedup only; excludes fitting, V4 generation, disk I/O and plotting.',
                selection='No feature matcher or final object classification; GT-best windows appear only as evaluation diagnostics.')
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,column,title in zip(axes,['mean_windows','conditional_coverage_95','compact_ratio_95_median'],
        ['Mean local windows','Coverage among coarse-covered GT','Median eligible window / GT area']):
        bars=ax.bar(summary.method,summary[column]);ax.bar_label(bars,fmt='%.2f');ax.set_title(title);ax.margins(y=.2)
    fig.suptitle(' / '.join(methods)+': rectangular ROI refinement')
    fig.savefig(output/'summary.png',dpi=150);plt.close(fig)
    if list(methods)==['FINAL']:
        base={tuple(shape) for shape in cfg['constrained_shapes']}
        extra=[shape for shape in combined_shapes(cfg) if tuple(shape) not in base]
        analysis=(f"FINAL keeps {len(base)} S2 shapes and adds {len(extra)} scale-boundary S1 shapes: {extra}. "
                  "Only global upper size bounds are rounded to the power-of-two grid; lower and aspect limits stay unchanged. "
                  "No perspective gate or image-specific GT controls generation. "
                  "This is a final candidate-generation configuration, not object recognition; feature matching remains pending. "
                  "Overlapping local alternatives are intentional. GT-best images are oracle diagnostics, not runtime selections.")
    else:
        analysis='Historical S1/S2/S3 comparison. FINAL is the selected S2-plus-S1 boundary fallback; feature matching is not evaluated.'
    (output/'analysis.md').write_text('# Geometry refinement development findings\n\n'+analysis+'\n')
    body=['<!doctype html><html lang="en"><meta charset="utf-8"><title>ROI refinement candidate generation</title><style>body{font:16px system-ui;margin:24px}img{width:100%}td,th{padding:7px}</style>',
          '<h1>Rectangular ROI refinement candidate generation</h1>',
          '<p>Frozen V4 green_blue_50 coarse ROIs. Rectangular windows only. GT is used after generation for evaluation, never to place windows. No intensity, color matching, HOG or classification is evaluated.</p>',
          '<p>Trapezoid/size priors are fitted on official training GT; this is in-sample development. Perspective is a camera-view assumption, not physical calibration. Windows intentionally overlap as alternatives; they have not been selected by a feature matcher.</p>',
          '<p>'+html.escape(analysis)+'</p>',
          summary.to_html(index=False,float_format=lambda v:f'{v:.3f}'),'<img src="summary.png">']
    body += [f'<h3>{html.escape(path)}</h3><img loading="lazy" src="images/{filename}">' for path,filename in gallery]
    (output/'index.html').write_text('\n'.join(body)+'</html>')
    print(summary.to_string(index=False),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path.home()/'Downloads/datasets/Roadwork_Cones_Dataset/roadwork_cones_dataset')
    parser.add_argument('--coarse',type=Path,default=Path(__file__).parents[1]/'ROI/v4/result_official')
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    parser.add_argument('--output',type=Path,help='Default: result_official; historical experiments: results.')
    parser.add_argument('--experiments',action='store_true',help='Run historical S1/S2/S3 instead of FINAL.')
    parser.add_argument('--fit-config',action='store_true')
    args=parser.parse_args()
    if args.fit_config:fit_config(args.data,args.config)
    methods=('S1','S2','S3') if args.experiments else ('FINAL',)
    output=args.output or Path(__file__).with_name('results' if args.experiments else 'result_official')
    evaluate(args.data,args.coarse,output,args.config,methods)
