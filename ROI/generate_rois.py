"""Generate reviewable RGB ROI crops from development images, without editing data."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PIL import Image

CLASSES = ('cone', 'sign')
GEOMETRIES = tuple((size, size*ratio) for size in (8,16,24,32,48,64) for ratio in (1,2))
BROAD_RULES = {
    'cone':dict(hue_min=5.,hue_max=55.,saturation_min=.25,value_min=.15,minimum_ratio=.02),
    'sign':dict(hue_min=5.,hue_max=70.,saturation_min=.20,value_min=.15,minimum_ratio=.03),
}


def windows(width, height):
    blocks=[]
    for bw,bh in GEOMETRIES:
        if bw>width or bh>height:continue
        xs=np.unique(np.r_[np.arange(0,width-bw+1,max(1,bw//2)),width-bw])
        ys=np.unique(np.r_[np.arange(0,height-bh+1,max(1,bh//2)),height-bh])
        x,y=np.meshgrid(xs,ys)
        blocks.append(np.column_stack([x.ravel(),y.ravel(),x.ravel()+bw,y.ravel()+bh]))
    return np.concatenate(blocks).astype(np.int32)


def fractions(mask, boxes):
    x0,y0,x1,y1=boxes.T
    integral=np.pad(mask.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
    count=integral[y1,x1]-integral[y0,x1]-integral[y1,x0]+integral[y0,x0]
    return count/((x1-x0)*(y1-y0))


def hsv_channels(rgb):
    hsv=np.array(Image.fromarray(rgb).convert('HSV'),dtype=float)
    return hsv[...,0]*360/256,hsv[...,1]/255,hsv[...,2]/255


def orange_masks(rgb, rules):
    h,s,v=hsv_channels(rgb)
    return {cls:((h>=rule['hue_min'])&(h<=rule['hue_max'])&
                 (s>=rule['saturation_min'])&(v>=rule['value_min'])) for cls,rule in rules.items()}


def overlap(box, boxes):
    boxes=np.asarray(boxes).reshape(-1,4)
    area=np.maximum(0,box[2:]-box[:2]).prod()
    other=np.maximum(0,boxes[:,2:]-boxes[:,:2]).prod(1)
    intersection=np.maximum(0,np.minimum(box[2:],boxes[:,2:])-np.maximum(box[:2],boxes[:,:2])).prod(1)
    return intersection/np.maximum(area+other-intersection,1e-30)


def read_development(project):
    data=Path(project)/'Test Dataset/validation'
    images=pd.read_csv(data/'images.csv')
    source=pd.read_csv(data/'annotations.csv')
    metadata=images.set_index('image');valid=[];invalid=[]
    for index,row in source.iterrows():
        box=np.array([row.x0,row.y0,row.x1,row.y1],dtype=float)
        ok=row.image in metadata.index and row['class'] in CLASSES and np.isfinite(box).all()
        if ok:
            m=metadata.loc[row.image]
            ok=0<=box[0]<box[2]<=m.width and 0<=box[1]<box[3]<=m.height
        item=dict(annotation_id=int(index),image=row.image,cls=row['class'],box=box.tolist())
        (valid if ok else invalid).append(item)
    return data,images,source,valid,invalid


def background_eligible(shape, boxes, source_rows):
    height,width=shape;occupied=np.zeros(shape,dtype=bool)
    for _,row in source_rows.iterrows():
        box=np.array([row.x0,row.y0,row.x1,row.y1],dtype=float)
        if not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:continue
        cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
        bw,bh=(box[2]-box[0])*1.25,(box[3]-box[1])*1.25
        x0,x1=np.clip([int(np.floor(cx-bw/2)),int(np.ceil(cx+bw/2))],0,width)
        y0,y1=np.clip([int(np.floor(cy-bh/2)),int(np.ceil(cy+bh/2))],0,height)
        occupied[y0:y1,x0:x1]=True
    return fractions(occupied,boxes)==0


def sample_background(boxes, eligible, broad_ratios, rng):
    """Keep the original 12 random-scale + 4 broad-color-hard samples per scene."""
    chosen=[];used=set()
    for width,height in GEOMETRIES:
        indices=np.flatnonzero(eligible&(boxes[:,2]-boxes[:,0]==width)&(boxes[:,3]-boxes[:,1]==height))
        if len(indices):
            index=int(rng.choice(indices));chosen.append((index,'random_scale_balanced'));used.add(index)
    score=np.maximum(broad_ratios['cone'],broad_ratios['sign'])
    indices=np.flatnonzero(eligible);tie=rng.random(len(indices))
    order=indices[np.lexsort((tie,-score[indices]))];hard=[]
    for index in order:
        if index in used or (hard and np.any(overlap(boxes[index],boxes[hard])>.3)):continue
        hard.append(int(index));chosen.append((int(index),'hard_color'));used.add(int(index))
        if len(hard)==4:break
    return chosen


def roi_records(image, rgb, annotations, boxes, sampled):
    rows=[]
    for number,(index,sampling) in enumerate(sampled):
        rows.append(dict(roi_id=f'bg_{Path(image).stem}_{number:02d}',image=image,label='background_candidate',
                         sampling=sampling,box=boxes[index].tolist()))
    for a in annotations:
        x0,y0,x1,y1=a['box']
        rows.append(dict(roi_id=f"obj_{a['annotation_id']:04d}",image=image,label=a['cls'],sampling='source_annotation',
                         box=[int(np.floor(x0)),int(np.floor(y0)),int(np.ceil(x1)),int(np.ceil(y1))]))
    return rows


def export(project, output):
    """Optional regeneration of actual crops; the clean report does not retain them."""
    data,images,source,annotations,_=read_development(project)
    output=Path(output);rng=np.random.default_rng(42);count=0
    for _,row in images.iterrows():
        with Image.open(data/row.image) as im:rgb=np.array(im.convert('RGB'))
        boxes=windows(rgb.shape[1],rgb.shape[0])
        ratios={c:fractions(mask,boxes) for c,mask in orange_masks(rgb,BROAD_RULES).items()}
        eligible=background_eligible(rgb.shape[:2],boxes,source[source.image==row.image])
        sampled=sample_background(boxes,eligible,ratios,rng)
        own=[a for a in annotations if a['image']==row.image]
        for record in roi_records(row.image,rgb,own,boxes,sampled):
            folder=output/record['label'];folder.mkdir(parents=True,exist_ok=True)
            x0,y0,x1,y1=record['box']
            Image.fromarray(rgb[y0:y1,x0:x1]).save(folder/(record['roi_id']+'.png'));count+=1
    print(f'Generated {count} unverified RGB ROI crops in {output}. Source datasets unchanged.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'roi_images')
    args=parser.parse_args();export(args.project,args.output)
