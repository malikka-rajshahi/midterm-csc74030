"""Rectangular local windows inside saved coarse ROIs; geometry only, no classes."""
from pathlib import Path
import json
import time
import numpy as np


def load_config(path=None):
    return json.loads(Path(path or Path(__file__).with_name('config.json')).read_text())


def combined_shapes(config):
    """S2 plus S1 upper-bound fallback aligned to the power-of-two size grid.

    Keep S2 lower limits and aspect intervals. Round only upper width/height
    limits upward to the next grid scale, avoiding quantile/grid boundary losses.
    This global rule never consults a test image or its GT.
    """
    selected={tuple(shape) for shape in config['constrained_shapes']}
    for w,h in config['generic_shapes']:
        for bounds in config['size_aspect_bounds']:
            upper_w=2**np.ceil(np.log2(bounds['width'][1]))
            upper_h=2**np.ceil(np.log2(bounds['height'][1]))
            if (bounds['width'][0]<=w<=upper_w and bounds['height'][0]<=h<=upper_h and
                    bounds['aspect'][0]<=w/h<=bounds['aspect'][1]):
                selected.add((w,h))
                break
    return sorted(selected)


def perspective_keep(boxes, config):
    """A fitted trapezoid limits bottom-center search points and plausible height."""
    boxes=np.asarray(boxes).reshape(-1,4)
    width,height=config['work_size']
    prior=config['perspective']
    x=(boxes[:,0]+boxes[:,2])/(2*width)
    y=boxes[:,3]/height
    t=(y-prior['top_y'])/(1-prior['top_y'])
    left=prior['top_left']+(prior['bottom_left']-prior['top_left'])*t
    right=prior['top_right']+(prior['bottom_right']-prior['top_right'])*t
    log_height=np.log(boxes[:,3]-boxes[:,1])
    expected=prior['intercept']+prior['slope']*y
    return ((y>=prior['top_y'])&(y<=1)&(x>=left)&(x<=right)&
            (log_height>=expected+prior['residual_low'])&
            (log_height<=expected+prior['residual_high']))


def generate_windows(coarse_boxes, config=None, method='FINAL'):
    """FINAL combines S2 and scale-boundary S1 fallback; historical S1/S2/S3 remain.

    Boxes use half-open working xyxy coordinates. All windows lie entirely within
    one coarse ROI. Sliding alternatives overlap intentionally; no classification
    score, image-specific GT or colors are consulted here.
    """
    started=time.perf_counter()
    config=load_config() if config is None else config
    if method not in {'S1','S2','S3','FINAL'}:
        raise ValueError('method must be FINAL, S1, S2 or S3.')
    boxes=np.asarray(coarse_boxes,dtype=float).reshape(-1,4)
    width,height=config['work_size']
    if not np.isfinite(boxes).all() or (boxes[:,:2]<0).any() or (boxes[:,2:]>[width,height]).any() or (boxes[:,2:]<=boxes[:,:2]).any():
        raise ValueError('Invalid working-coordinate coarse ROIs.')
    # Inward rounding guarantees containment even for nonintegral caller inputs.
    boxes=np.c_[np.ceil(boxes[:,:2]),np.floor(boxes[:,2:])].astype(int)
    bank=(combined_shapes(config) if method=='FINAL' else
          config['generic_shapes'] if method=='S1' else config['constrained_shapes'])
    if any(len(shape)!=2 or any(v<=0 or int(v)!=v for v in shape) for shape in bank):
        raise ValueError('Window dimensions must be positive integers.')
    stride=config['stride_fraction']
    if not 0<stride<=1:
        raise ValueError('stride_fraction must be in (0,1].')
    blocks=[]
    for x0,y0,x1,y1 in boxes:
        for bw,bh in bank:
            if bw>x1-x0 or bh>y1-y0:
                continue
            xs=np.unique(np.r_[np.arange(x0,x1-bw+1,max(1,round(bw*stride))),x1-bw])
            ys=np.unique(np.r_[np.arange(y0,y1-bh+1,max(1,round(bh*stride))),y1-bh])
            x,y=np.meshgrid(xs,ys)
            candidate=np.c_[x.ravel(),y.ravel(),x.ravel()+bw,y.ravel()+bh]
            if method=='S3':
                candidate=candidate[perspective_keep(candidate,config)]
            blocks.append(candidate)
    windows=np.unique(np.concatenate(blocks).astype(np.int32),axis=0) if blocks else np.empty((0,4),np.int32)
    return dict(boxes=windows,method=method,runtime_seconds=time.perf_counter()-started)
