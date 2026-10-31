"""Run the fixed A/B/C/D experiment and generate the image report and README."""
from pathlib import Path
import argparse
import html
import json
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from generate_rois import (CLASSES,BROAD_RULES,windows,fractions,hsv_channels,orange_masks,
                           overlap,read_development,background_eligible,sample_background,roi_records)

# Frozen existing grayscale classifier: do not refit it when comparing prescreens.
REFERENCES={'cone':1708.3979316032287,'sign':952.3059516320549}
THRESHOLDS={'cone':682.2053792730977,'sign':-682.2053792730977}
D_RULES={
 'cone':dict(hue_min=10.,hue_max=40.,saturation_min=.20,value_min=.20,
             orange_min=.029,white_min=.01,combined_min=.058),
 'sign':dict(hue_min=10.,hue_max=40.,saturation_min=.25,value_min=.20,
             orange_min=.048,white_min=0.,combined_min=0.),
}
GEOMETRY={'cone':(1.2,4.0),'sign':(.5,1.8)}
NAMES={'A':'Current baseline','B':'Broad HSV ratio','C':'Broad HSV + geometry','D':'Mask-derived orange/white + geometry'}


def variance_in_boxes(gray, boxes):
    x0,y0,x1,y1=boxes.T;area=(x1-x0)*(y1-y0);moments=[]
    for power in (1,2):
        integral=np.pad(gray.astype(float)**power,((1,0),(1,0))).cumsum(0).cumsum(1)
        moments.append((integral[y1,x1]-integral[y0,x1]-integral[y1,x0]+integral[y0,x0])/area)
    return np.maximum(0,moments[1]-moments[0]**2)


def d_components(rgb):
    h,s,v=hsv_channels(rgb)
    yellow=(h>=45)&(h<=80)&(s>=.10)&(v>=.15)
    white=(s<=.15)&(v>=.65)&~yellow
    orange={c:mask&~yellow for c,mask in orange_masks(rgb,D_RULES).items()}
    return orange,white


def color_ratios(rgb, boxes):
    broad={c:fractions(mask,boxes) for c,mask in orange_masks(rgb,BROAD_RULES).items()}
    orange,white=d_components(rgb)
    focused={c:fractions(mask,boxes) for c,mask in orange.items()}
    focused['white']=fractions(white,boxes)
    for cls in CLASSES:focused[cls+'_combined']=fractions(orange[cls]|white,boxes)
    return broad,focused


def prescreen(boxes,broad,focused,cls,experiment):
    if experiment=='A':return np.ones(len(boxes),dtype=bool)
    if experiment in ['B','C']:
        keep=broad[cls]>=BROAD_RULES[cls]['minimum_ratio']
    else:
        rule=D_RULES[cls];keep=focused[cls]>=rule['orange_min']
        if cls=='cone':
            keep&=(focused['white']>=rule['white_min'])&(focused[cls+'_combined']>=rule['combined_min'])
    if experiment in ['C','D']:
        ratio=(boxes[:,3]-boxes[:,1])/(boxes[:,2]-boxes[:,0]);lo,hi=GEOMETRY[cls]
        keep&=(ratio>=lo)&(ratio<=hi)
    return keep


def nms(boxes,indices):
    kept=[];remaining=indices
    while len(remaining):
        index=int(remaining[0]);kept.append(index);rest=remaining[1:]
        remaining=rest[overlap(boxes[index],boxes[rest])<=.3]
    return kept[:100]


def predict(boxes,values,broad,focused,experiment):
    distances={c:np.abs(values-REFERENCES[c]) for c in CLASSES};predictions=[]
    for cls in CLASSES:
        other=next(c for c in CLASSES if c!=cls)
        margin=distances[other]-distances[cls]
        allowed=prescreen(boxes,broad,focused,cls,experiment)&(margin>=THRESHOLDS[cls])
        ids=np.flatnonzero(allowed);ranked=ids[np.lexsort((ids,distances[cls][ids]))][:1000]
        for index in nms(boxes,ranked):
            predictions.append(dict(cls=cls,box=boxes[index].tolist(),distance=float(distances[cls][index])))
    return predictions


def match(predictions,annotations):
    used=set();matches={}
    for index,pred in enumerate(predictions):
        options=[i for i,a in enumerate(annotations) if i not in used and a['cls']==pred['cls']]
        if not options:continue
        values=overlap(np.array(pred['box']),[annotations[i]['box'] for i in options]);best=int(np.argmax(values))
        if values[best]>=.5:
            target=options[best];used.add(target);matches[index]=target
    return matches,set(range(len(annotations)))-used


def annotate(ax,box,color,dashed=False,label=''):
    x0,y0,x1,y1=box
    ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=color,linewidth=1.2,linestyle='--' if dashed else '-'))
    if label:ax.text(x0,y0,label,color=color,fontsize=6,bbox=dict(facecolor='black',alpha=.4,pad=.3,edgecolor='none'))


def scene_report(rgb,annotations,predictions,path,image):
    fig,axes=plt.subplots(1,5,figsize=(28,6.3),layout='constrained')
    for ax in axes:ax.imshow(rgb);ax.axis('off')
    for a in annotations:annotate(axes[0],a['box'],'cyan',True,a['cls'])
    axes[0].set_title(f'Source labels (unverified): {len(annotations)}',fontsize=10)
    for ax,e in zip(axes[1:],NAMES):
        matches,missed=match(predictions[e],annotations)
        for i,p in enumerate(predictions[e]):annotate(ax,p['box'],'lime' if i in matches else 'red')
        for i in missed:annotate(ax,annotations[i]['box'],'orange',True)
        ax.set_title(f'{e}: {NAMES[e]}\nMatched {len(matches)} | Outputs {len(predictions[e])} | Unmatched labels {len(missed)}',fontsize=9)
    fig.suptitle(f'{image}\nCyan: source labels | Green: matched prediction | Red: unmatched prediction | Orange dashed: unmatched label\nSame image, same candidate bank and classifier. Labels are unverified; unmatched does not prove a model error.',fontsize=11)
    fig.savefig(path,dpi=120);plt.close(fig)


def score(tp,fp,fn):
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
    return dict(tp=tp,fp=fp,fn=fn,precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0.)


def run(project,output,verify_sources=None):
    project,output=Path(project),Path(output);data,images,source,annotations,invalid=read_development(project)
    (output/'results/images').mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(42);counts={e:{c:np.zeros(3,dtype=int) for c in CLASSES} for e in NAMES}
    roi_stats=[];scene_stats=[];roi_identity=[]
    previous={e:json.loads(Path(path).read_text()) for e,path in (verify_sources or {}).items()}
    for number,row in images.iterrows():
        with Image.open(data/row.image) as im:rgb=np.array(im.convert('RGB'))
        gray=np.array(Image.fromarray(rgb).convert('L'));boxes=windows(gray.shape[1],gray.shape[0]);values=variance_in_boxes(gray,boxes)
        broad,focused=color_ratios(rgb,boxes);own=[a for a in annotations if a['image']==row.image]
        predictions={e:predict(boxes,values,broad,focused,e) for e in NAMES}
        for e,rows in predictions.items():
            if e in previous:
                expected=[dict(cls=p['class'],box=p['box'],distance=p['reference_distance']) for p in previous[e][row.image]]
                if rows!=expected:raise AssertionError(f'{e} differs from its saved result: {row.image}')
            matches,missed=match(rows,own)
            for cls in CLASSES:
                tp=sum(rows[i]['cls']==cls for i in matches);total=sum(p['cls']==cls for p in rows);gt=sum(a['cls']==cls for a in own)
                counts[e][cls]+=np.array([tp,total-tp,gt-tp])
            scene_stats.append(dict(image=row.image,experiment=e,matched=len(matches),predictions=len(rows),unmatched_labels=len(missed),plot=f'images/{number:03d}_{Path(row.image).stem}.png'))
        eligible=background_eligible(gray.shape,boxes,source[source.image==row.image])
        sampled=sample_background(boxes,eligible,broad,rng)
        records=roi_records(row.image,rgb,own,boxes,sampled);roi_identity.extend(records)
        roi_boxes=np.array([item['box'] for item in records]);rb,rf=color_ratios(rgb,roi_boxes);rv=variance_in_boxes(gray,roi_boxes)
        for e in NAMES:
            passed={c:prescreen(roi_boxes,rb,rf,c,e) for c in CLASSES}
            accepted={}
            for cls in CLASSES:
                other=next(c for c in CLASSES if c!=cls)
                margin=np.abs(rv-REFERENCES[other])-np.abs(rv-REFERENCES[cls])
                accepted[cls]=passed[cls]&(margin>=THRESHOLDS[cls])
            for i,item in enumerate(records):
                label=item['label'];roi_stats.append(dict(experiment=e,label=label,sampling=item['sampling'],
                    pre=bool(passed[label][i]) if label in CLASSES else bool(passed['cone'][i] or passed['sign'][i]),
                    post=bool(accepted[label][i]) if label in CLASSES else bool(accepted['cone'][i] or accepted['sign'][i])))
        scene_report(rgb,own,predictions,output/'results/images'/f'{number:03d}_{Path(row.image).stem}.png',row.image)
        if number%10==0:print(f'{number+1}/{len(images)} ABCD scenes complete',flush=True)
    metrics=[]
    for e in NAMES:
        for cls in [*CLASSES,'all']:
            sums=counts[e][cls] if cls!='all' else sum(counts[e].values())
            metrics.append(dict(experiment=e,cls=cls,**score(*map(int,sums))))
    result=dict(metrics=pd.DataFrame(metrics),rois=pd.DataFrame(roi_stats),scenes=pd.DataFrame(scene_stats),
                roi_identity=roi_identity,valid_rows=len(annotations),invalid_rows=len(invalid))
    write_report(output,result)
    print(result['metrics'].to_string(index=False));return result


def write_report(output,result):
    metrics=result['metrics'];rois=result['rois'];scenes=result['scenes'];overall=metrics[metrics.cls=='all'].set_index('experiment')
    roi_table=[]
    for e in NAMES:
        own=rois[rois.experiment==e]
        values=dict(experiment=e)
        for group,key in [('random_scale_balanced','random_rejection'),('hard_color','hard_rejection')]:
            values[key]=1-float(own[(own.label=='background_candidate')&(own.sampling==group)].pre.mean())
        values['cone_retention']=float(own[own.label=='cone'].pre.mean())
        values['sign_retention']=float(own[own.label=='sign'].pre.mean())
        roi_table.append(values)
    roi_table=pd.DataFrame(roi_table).set_index('experiment')
    macro=metrics[metrics.cls.isin(CLASSES)].groupby('experiment').f1.mean();winner=str(macro.idxmax())
    # One exportable image summarizes all four methods and their tradeoffs.
    fig=plt.figure(figsize=(18,11),layout='constrained');grid=fig.add_gridspec(3,3,height_ratios=[1.1,1,1.1])
    ax=fig.add_subplot(grid[0,:]);ax.axis('off')
    table=[]
    for e in NAMES:
        r=overall.loc[e]
        table.append([e,NAMES[e],int(r.tp),int(r.tp+r.fp),int(r.fn),f'{r.precision*100:.4f}%',f'{r.recall*100:.3f}%',f'{r.f1*100:.4f}%'])
    artist=ax.table(cellText=table,colLabels=['Exp','Prescreen','Matched','Outputs','Unmatched labels','Precision*','Recall*','F1*'],loc='center',cellLoc='center',colWidths=[.045,.30,.075,.09,.12,.09,.09,.09])
    artist.auto_set_font_size(False);artist.set_fontsize(10);artist.scale(1,2.2)
    for (row,col),cell in artist.get_celld().items():
        if row==0:cell.set_facecolor('#183650');cell.set_text_props(color='white',weight='bold')
        elif row%2==0:cell.set_facecolor('#edf3f7')
    colors=['#8d99ae','#457b9d','#2a9d8f','#e9a23b']
    for col,(field,title) in enumerate([('tp','Matched source rows'),('recall','Provisional recall (%)'),('f1','Provisional F1 (%)')]):
        ax=fig.add_subplot(grid[1,col]);values=overall[field].to_numpy()*(100 if field!='tp' else 1)
        bars=ax.bar(list(NAMES),values,color=colors);ax.set_title(title);ax.spines[['top','right']].set_visible(False)
        for b,value in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,b.get_height(),f'{value:.4f}' if field=='f1' else f'{value:.2f}',ha='center',va='bottom',fontsize=9)
        ax.margins(y=.2)
    for col,(fields,title) in enumerate([(['random_rejection','hard_rejection'],'Initial-gate background rejection (%)'),(['cone_retention','sign_retention'],'Initial-gate labeled-object retention (%)')]):
        ax=fig.add_subplot(grid[2,col]);roi_table[fields].mul(100).plot.bar(ax=ax,rot=0,color=['#457b9d','#e9a23b'])
        ax.set(title=title,ylim=(0,110),xlabel='Experiment');ax.legend(['Random background','Color-hard background'] if col==0 else ['Cone','Sign'],fontsize=8)
    ax=fig.add_subplot(grid[2,2]);ax.axis('off')
    ax.text(0,.98,f'INTERPRETATION\n\n{winner} has the highest provisional macro-F1.\n\nD better matches the orange/white target\ndefinition and rejects more sampled\nbackground, but retains fewer labeled\nobjects and does not improve total F1.\n\nOnly development data is used.\nSource labels and background samples\nare unverified. * Metrics are provisional.',va='top',fontsize=11,linespacing=1.5)
    fig.suptitle('A / B / C / D — controlled development comparison',fontsize=20,weight='bold')
    fig.savefig(output/'results/summary.png',dpi=150);plt.close(fig)
    # Direct image gallery: no predictions caches, review tables or analysis helper files.
    body=['<!doctype html><html lang="en"><meta charset="utf-8"><title>A/B/C/D image report</title><style>body{font:16px system-ui;margin:24px;background:#f5f7fa;color:#17293a}img{width:100%;background:white}article{background:white;border-radius:8px;padding:16px;margin:20px 0}summary{cursor:pointer}.note{background:#fff1c5;padding:16px}</style><h1>A / B / C / D image report</h1><p class="note">Development only. Source labels are unverified. Red predictions and orange labels are unmatched, not confirmed model errors. All retained predictions are shown.</p><p><a href="../README.md">Results and full analysis</a> · <a href="summary.png">Open summary image</a></p><img src="summary.png" alt="ABCD results and analysis"><h2>All 120 scenes: supplied labels + A + B + C + D</h2>']
    for _,row in scenes[scenes.experiment=='A'].iterrows():
        stats=[]
        for e in NAMES:
            r=scenes[(scenes.image==row.image)&(scenes.experiment==e)].iloc[0]
            stats.append(f'{e}: {int(r.matched)} matched / {int(r.predictions)} outputs')
        body.append(f'<article><details><summary>{html.escape(row.image)} — {" | ".join(stats)}</summary><a href="{row["plot"]}"><img loading="lazy" src="{row["plot"]}" alt="Source labels and four experiment predictions"></a></details></article>')
    body.append('</html>');(output/'results/index.html').write_text('\n'.join(body))
    # This is both the results document and the concise reproduction guide.
    lines=['# ROI experiments A / B / C / D','','## Start with the image report','','![Four-experiment summary](results/summary.png)','','Open [all 120 same-scene comparisons](results/index.html). Each image shows **source labels + A + B + C + D**, in that order. Cyan = supplied labels; green = class-matched one-to-one prediction; red = unmatched prediction; orange dashed = unmatched supplied label. Every retained output is drawn; no additional display cap is used.','','## What each experiment means','','| Experiment | Initial filter | Geometry filter |','|---|---|---|','| A | Existing current baseline; no background rejection | None |','| B | Original broad HSV color ratio | None |','| C | Original broad HSV color ratio | Cone/sign height-to-width ranges |','| D | Latest reference-mask-derived orange/white rules | Same geometry as C |','','A/B/C retain their original definitions. **D is the latest orange/white-plus-geometry configuration**, previously stored as C in the later run.','','All four share the same candidate windows, frozen grayscale Variance references and class thresholds, ranking, NMS and output limits. This isolates the effect of the initial filters. RGB is used for HSV; grayscale population variance is used afterward.','','## Numerical results: whole scenes','','These are provisional class-matched IoU >= 0.5 results against the original, uncorrected annotation rows.','', '| Exp | Matched | Unmatched predictions | Unmatched labels | Precision | Recall | F1 |','|---|---:|---:|---:|---:|---:|---:|']
    for e in NAMES:
        r=overall.loc[e];lines.append(f'| {e} | {int(r.tp)} | {int(r.fp)} | {int(r.fn)} | {r.precision*100:.4f}% | {r.recall*100:.3f}% | {r.f1*100:.4f}% |')
    lines+=['','### Cone and sign separately','','| Exp | Class | Matched | Unmatched predictions | Unmatched labels | Precision | Recall | F1 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for _,r in metrics[metrics.cls.isin(CLASSES)].iterrows():lines.append(f'| {r.experiment} | {r.cls} | {r.tp} | {r.fp} | {r.fn} | {r.precision*100:.4f}% | {r.recall*100:.3f}% | {r.f1*100:.4f}% |')
    lines+=['','## Initial-filter behavior: rejection versus retention','','These statistics describe the prescreen alone, before the unchanged Variance gate. Background rejection uses union acceptance across both classes. Object retention tests the supplied object class. The same 1,920 background candidate ROIs and 228 supplied object ROIs are regenerated for all four methods.','','| Exp | Random background rejected | Hard-color background rejected | Cone ROIs retained | Sign ROIs retained |','|---|---:|---:|---:|---:|']
    for e,r in roi_table.iterrows():lines.append(f'| {e} | {r.random_rejection*100:.2f}% | {r.hard_rejection*100:.2f}% | {r.cone_retention*100:.2f}% | {r.sign_retention*100:.2f}% |')
    lines+=['','## Analysis: what the evidence supports','','### A: the current classifier has no background rejection','','A retains every sampled background candidate and produces 24,000 outputs with no source-label matches. Its grayscale class margin compares cone with sign, not object with background. A road, wall or building region can therefore be assigned to one of the two classes. The result identifies a structural limitation: intensity-based cone/sign discrimination alone is not a foreground detector.','','### B: color removes many easy backgrounds but struggles with similar colors','','B rejects about 85.35% of the randomly sampled background candidates, but only 0.21% of the deliberately color-hard candidates. Thus the loose HSV rule is useful for rejecting many ordinary regions, yet it provides almost no protection against orange/yellow distractors. B matches two source rows, which is a small development gain over A rather than evidence of a usable detector.','','The broad old hue ranges intentionally permitted yellow-orange, and the sign range also admitted yellow. Therefore some accepted regions satisfy the numeric rule without matching the intended orange construction-object definition.','','### C: geometry improves this comparison, with a retention cost','','Compared with B, C increases matched source rows from two to four and has the highest provisional macro-F1 among these fixed configurations. Cone matching improves from one to three; sign matching remains one. The geometry filter changes which candidates survive ranking/NMS, so final outputs need not be subsets of the B outputs.','','However, sign prescreen retention falls from 60.53% to 52.63%. The unchanged candidate bank contains only square and 2:1-tall windows; C keeps tall windows for cone and square windows for sign. This is a restrictive shape approximation and can reject a genuine target whose visible shape or annotation box differs. C is only the best score on this small, noisy development comparison, not an established universal winner.','','### D: removing yellow fits the intended target, but the current rules are stricter','','D excludes yellow from positive orange/white evidence and separates the classes: cone requires orange and white; sign requires orange only. This better matches the requested color definition and rejects more sampled backgrounds: random-background rejection rises from C\'s 87.08% to D\'s 95.00%, and hard-color rejection rises from 2.29% to 47.71%.','','**Removing yellow is consistent with the target definition, but the current rule is stricter.** In particular, requiring cone white coverage can reject cones with very little visible white, white stripes darkened by lighting, or solid-orange construction cones. The full-ROI denominator also dilutes orange/white fractions when a candidate contains more background. Conversely, orange and white pixels inside the same ROI need not belong to the same object; a wall and sky can provide both colors without forming a cone. These gates do not perform object segmentation.','','The tradeoff is visible in the source-object ROI retention: cone falls from 65.13% in C to 32.24% in D, and sign from 52.63% to 28.95%. Scene matches fall from four to three; provisional total F1 also decreases. Therefore D is better at rejecting these sampled backgrounds, but **does not improve overall detection in this run**.','','This is not proof that white alone caused the decline. C-to-D changes hue limits, saturation/value floors, coverage thresholds and the cone-white requirement together. A controlled white-threshold ablation would be required to isolate that effect. No such sweep is included in these four results.','','### Annotation uncertainty limits every conclusion','','Source annotations are uncorrected model outputs. Duplicate rows can inflate unmatched-label counts; missing labels can make a real target appear to be an unmatched prediction; inaccurate box boundaries and wrong classes can cause IoU/class mismatches. Source sign crops also include at least one stop-sign-like example, which may conflict with the orange temporary-construction-sign reference definition. These are review issues, not automatically corrected labels.','','Consequently, red boxes are not confirmed false positives and orange boxes are not confirmed missed real objects. All reported precision/recall/F1 values are provisional. Four versus three matches is weak evidence when labels are uncertain. The most useful current conclusion is the observed background-rejection/target-retention tradeoff, not a claim that one method is ready for deployment.','','## Exact rules and shared settings','','### B/C broad HSV rules','','- Cone: hue 5–55 degrees, saturation >=0.25, value >=0.15; compatible pixels >=2% of ROI.','- Sign: hue 5–70 degrees, saturation >=0.20, value >=0.15; compatible pixels >=3% of ROI.','- Hue is PIL HSV H * 360/256; saturation and value are divided by 255.','','### D separate orange/white rules','','- Both orange hue ranges: 10–40 degrees.','- Cone orange: S>=0.20, V>=0.20, orange coverage >=2.9%.','- Cone white: S<=0.15, V>=0.65, white coverage >=1.0%; orange+white coverage >=5.8%. All three coverage conditions must pass.','- Sign orange: S>=0.25, V>=0.20, orange coverage >=4.8%. No white or combined-coverage condition.','- Yellow H 45–80 degrees with S>=0.10 and V>=0.15 is excluded from both orange and white evidence.','- All coverage denominators are the entire ROI. White alone never passes. An ROI can contain yellow background, but yellow provides no positive evidence.','','D\'s orange/white distributions and coverage thresholds were derived from the original 183 cone / 36 sign reference masks only, using source-balanced object distributions and fixed extra-background allowances. Hue was bounded by an explicit orange anchor, not automatically allowed to learn yellow as orange. The 1% cone-white minimum is the user-requested adjustable condition, not a universal property learned for all cones. See `D_RULES` in `run_experiments.py` to change these controls for a new development experiment; changing them redefines D and requires rerunning the report.','','### Geometry and common classifier','','- Geometry in C/D only: cone H/W 1.2–4.0; sign H/W 0.5–1.8.','- Candidates: widths 8,16,24,32,48,64 pixels; height/width 1 or 2; half strides, edge-aligned final windows.','- Frozen Variance references: cone 1708.3979316032287; sign 952.3059516320549.','- Class margin: distance to other reference minus distance to own reference. Thresholds: cone +682.2053792730977; sign -682.2053792730977.','- Rank accepted candidates by own-class reference distance, with original index as tie-break; keep top 1,000 before NMS.','- NMS IoU 0.3; maximum 100 outputs per class. Matching: class-specific, one-to-one IoU >=0.5.','','## Data use and ROI sampling','','Only `Test Dataset/validation` is evaluated: 120 images, 228 valid original annotation rows and one invalid row. Original labels and images are unchanged. Neither NYC test nor final-test images under `Downloads/datasets` are read. These results are development comparisons, not a final benchmark.','','The ROI set is deterministic (seed 42): one random crop for each of 12 window geometries per image (1,440 total), plus four color-hard crops per image (480 total). Sampling excludes every supplied positive-area box expanded by 1.25x. Hard samples are always selected by the original broad HSV score, including for D; the set is not reselected to favor new rules. Missing labels may still contaminate background candidates, and the deliberately hard sampling is not representative of scene prevalence.','','## Minimal folder and reproduction','','Only the actual generation code, this README and image-report outputs are retained. Old notebooks, caches, configs, manifests, prediction dumps, review helpers and experiment backups have been removed. Numerical results are embedded in this README and the summary image. Original datasets and Intensity main code are preserved.','','```text','ROI/','  generate_rois.py       # shared ROI generation and optional crop export','  run_experiments.py     # frozen A/B/C/D inference + image report + README','  README.md','  results/','    summary.png','    index.html','    images/             # all 120 five-panel comparison images','```','','Dependencies: numpy, pandas, pillow, matplotlib. From the project root:','','```sh','python ROI/run_experiments.py','```','','To regenerate actual RGB ROI crops when needed (not stored in the clean default package):','','```sh','python ROI/generate_rois.py --output ROI/roi_images','```','','Both commands default to the parent project directory and use the validation split only. ROI export creates 1,920 background-candidate and 228 source-label crops; all labels remain unverified. No extra cached result files are required to reproduce the report.','']
    (output/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();run(args.project,args.output)
