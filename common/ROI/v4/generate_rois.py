"""ROI V4: red/orange anchors, training-sized windows, background gate, group closure."""
from pathlib import Path
import json
import time
import cv2
import numpy as np
from common.preprocessing import denoise, to_hsv
from .sliding_window import integral, fractions, anchored_windows, merge_overlaps


def load_config(path=None):
    return json.loads(Path(path or Path(__file__).with_name('roi_config.json')).read_text())


def color_evidence(image_bgr, saturation_min=0.35, value_min=0.15):
    """Degree-based HSV evidence; only the red/orange union can seed proposals."""
    hsv=to_hsv(image_bgr)
    hue=hsv[...,0].astype(float)*2
    sat=hsv[...,1]/255.
    value=hsv[...,2]/255.
    chromatic=(sat>=saturation_min)&(value>=value_min)
    bands={'red':((0,15),(345,360)), 'orange':((15,45),),
           'yellow':((45,65),), 'green':((65,170),), 'blue':((190,260),)}
    masks={name: np.logical_or.reduce([(hue>=lo)&(hue<hi) for lo,hi in ranges])&chromatic
           for name,ranges in bands.items()}
    masks['white']=(sat<=.18)&(value>=.60)
    masks['support']=masks['red']|masks['orange']
    return masks


def find_anchors(support, block_sizes=(2,4,8), min_pixels=4):
    """Group one support mask at several block granularities, then tighten to support.

    Tightening affects anchors only. Final ROIs retain context and all pixel colors.
    """
    height,width=support.shape
    anchors=set()
    for size in block_sizes:
        rows,cols=(height+size-1)//size,(width+size-1)//size
        padded=np.pad(support,((0,rows*size-height),(0,cols*size-width)))
        counts=padded.reshape(rows,size,cols,size).sum(axis=(1,3))
        area=np.minimum(size,height-np.arange(rows)*size)[:,None]*np.minimum(size,width-np.arange(cols)*size)
        active=((counts>=2)&(counts/area>=.15)).astype(np.uint8)
        _,_,stats,_=cv2.connectedComponentsWithStats(active,connectivity=8)
        for x,y,w,h,_ in stats[1:]:
            x0,y0=x*size,y*size
            x1,y1=min(width,(x+w)*size),min(height,(y+h)*size)
            yy,xx=np.nonzero(support[y0:y1,x0:x1])
            if len(xx)>=min_pixels:
                anchors.add((x0+int(xx.min()),y0+int(yy.min()),x0+int(xx.max())+1,y0+int(yy.max())+1))
    return np.array(sorted(anchors),dtype=np.int32).reshape(-1,4)


def generate_proposals(image_bgr, *, config=None, apply_denoise=True,
                       return_masks=False):
    """Return original-coordinate boxes plus explicit working-coordinate artifacts.

    No image-specific GT, class decision, feature model or arbitrary output cap.
    Selected green_blue_50 pipeline. Color fractions use whole boxes.
    """
    started=time.perf_counter()
    if not isinstance(image_bgr,np.ndarray) or image_bgr.dtype!=np.uint8 or image_bgr.ndim!=3 or image_bgr.shape[2]!=3 or min(image_bgr.shape[:2])<1:
        raise ValueError('Expected a nonempty uint8 BGR image.')
    cfg=load_config() if config is None else config
    ww,wh=cfg['work_size']
    original_h,original_w=image_bgr.shape[:2]
    work=cv2.resize(image_bgr,(ww,wh),interpolation=cv2.INTER_AREA)
    if apply_denoise:
        work=denoise(work)
    masks=color_evidence(work,cfg['saturation_min'],cfg['value_min'])
    anchors=find_anchors(masks['support'],tuple(cfg['anchor_blocks']),cfg['min_anchor_pixels'])
    boxes,sources=anchored_windows(anchors,cfg['window_sizes'],work.shape,
                                  padding=cfg['anchor_padding'],nearest=cfg['nearest_windows'],
                                  max_anchor_area_ratio=cfg['window_anchor_area_ratio'])
    integrals={name:integral(mask) for name,mask in masks.items()}
    def background_ok(candidates):
        return fractions(candidates,integrals['green']+integrals['blue'])<=.5
    generated=len(boxes)
    area=np.prod(boxes[:,2:]-boxes[:,:2],axis=1)
    area_ok=area<=ww*wh*cfg['max_roi_area_fraction']
    color_ok=background_ok(boxes)
    keep=area_ok&color_ok
    boxes=boxes[keep]
    sources=[ids for ids,ok in zip(sources,keep) if ok]
    before_merge=len(boxes)
    boxes,sources=merge_overlaps(boxes,sources,background_ok)
    evidence=[dict(color_ratios={name:float(fractions(box[None],summed)[0]) for name,summed in integrals.items()},
                   anchor_ids=ids,oversized=bool(np.prod(box[2:]-box[:2])>ww*wh*cfg['max_roi_area_fraction']),area_fraction=float(np.prod(box[2:]-box[:2])/(ww*wh)))
              for box,ids in zip(boxes,sources)]
    mapped=boxes.astype(float)*[original_w/ww,original_h/wh,original_w/ww,original_h/wh]
    mapped[:,:2]=np.floor(mapped[:,:2])
    mapped[:,2:]=np.ceil(mapped[:,2:])
    elapsed=time.perf_counter()-started
    return dict(boxes=mapped.astype(np.int32),work_boxes=boxes,image_bgr=image_bgr.copy(),
        work_image_bgr=work,evidence=evidence,anchors=anchors,
        color_masks=masks if return_masks else None,work_size=[ww,wh],
        counts=dict(anchors=len(anchors),generated=generated,area_rejected=int((~area_ok).sum()),
                    background_rejected=int((area_ok&~color_ok).sum()),before_merge=before_merge,final=len(boxes),
                    oversized=sum(info['oversized'] for info in evidence)),
        runtime_seconds=elapsed,background_mode='green_blue_50')
