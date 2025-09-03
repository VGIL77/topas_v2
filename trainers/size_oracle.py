
# Size Oracle Module
# Enhanced, robust size inference for ARC tasks with confidence scoring.
# Supports affine transforms, ratio preservation, bbox analysis, pattern-based inference
from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import torch
from fractions import Fraction
import math
import logging

logger = logging.getLogger(__name__)

def _shape_of(grid) -> Tuple[int,int]:
    # Accept numpy or torch tensors
    try:
        import torch
        if isinstance(grid, torch.Tensor):
            if grid.dim() == 3:  # [B,H,W] or [1,H,W]
                return int(grid.shape[-2]), int(grid.shape[-1])
            elif grid.dim() == 2:
                return int(grid.shape[0]), int(grid.shape[1])
    except Exception:
        pass
    a = np.asarray(grid)
    if a.ndim == 3:
        return int(a.shape[-2]), int(a.shape[-1])
    return int(a.shape[0]), int(a.shape[1])

def _bbox_nonzero(grid) -> Tuple[int,int,int,int]:
    # Returns (y0,x0,y1,x1) inclusive bbox over nonzero pixels; if all zero, whole grid
    g = grid
    try:
        import torch
        if isinstance(grid, torch.Tensor):
            g = grid.detach().cpu().numpy()
    except Exception:
        pass
    g = np.asarray(g)
    if g.ndim == 3:
        g = g[0]
    ys, xs = np.nonzero(g != 0)
    if ys.size == 0:
        H, W = g.shape
        return 0, 0, H-1, W-1
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())

def _fit_affine_int(demos_hw: List[Tuple[Tuple[int,int], Tuple[int,int]]],
                    coef_range = range(-2,3), bias_range = range(-5,6)) -> Optional[Tuple[int,int,int,int,int,int]]:
    """
    Fit small-integer affine map:
      Hout = a1*Hin + b1*Win + c1
      Wout = a2*Hin + b2*Win + c2
    Returns tuple (a1,b1,c1,a2,b2,c2) if exact for all demos, else None.
    """
    if not demos_hw:
        return None
    Hin_vals = [h for (h,w),(_H,_W) in demos_hw]
    Win_vals = [w for (h,w),(_H,_W) in demos_hw]
    Hout_vals= [_H for (h,w),(_H,_W) in demos_hw]
    Wout_vals= [_W for (h,w),(_H,_W) in demos_hw]

    # Quick shortcut: if all outputs identical size, return constant map
    if len(set(Hout_vals)) == 1 and len(set(Wout_vals)) == 1:
        c1 = Hout_vals[0]; c2 = Wout_vals[0]
        return (0,0,c1, 0,0,c2)

    # Enumerate small integer coefficients
    for a1 in coef_range:
        for b1 in coef_range:
            for c1 in bias_range:
                okH = all(a1*hin + b1*win + c1 == hout
                          for (hin,win), (hout,_) in demos_hw)
                if not okH: 
                    continue
                for a2 in coef_range:
                    for b2 in coef_range:
                        for c2 in bias_range:
                            okW = all(a2*hin + b2*win + c2 == wout
                                      for (hin,win), (_,wout) in demos_hw)
                            if okW:
                                return (a1,b1,c1,a2,b2,c2)
    return None

def _ratio_rational(x: float, max_den:int=12) -> Fraction:
    try:
        return Fraction(x).limit_denominator(max_den)
    except Exception:
        # Fallback: approximate
        num = int(round(x * max_den))
        return Fraction(num, max_den) if max_den>0 else Fraction(1,1)

def _fit_axiswise_ratio(demos_hw: List[Tuple[Tuple[int,int], Tuple[int,int]]], max_den:int=12) -> Optional[Tuple[Fraction, Fraction]]:
    """Fit H_out = rH * H_in, W_out = rW * W_in with small rational ratios."""
    if not demos_hw: return None
    ratios_H = []
    ratios_W = []
    for (hin,win), (hout,wout) in demos_hw:
        if hin<=0 or win<=0: return None
        ratios_H.append(hout/float(hin))
        ratios_W.append(wout/float(win))
    # Use median ratio, then rationalize
    rH = sorted(ratios_H)[len(ratios_H)//2]
    rW = sorted(ratios_W)[len(ratios_W)//2]
    return (_ratio_rational(rH, max_den), _ratio_rational(rW, max_den))

def _consistent_bbox_mapping(demos) -> Optional[Tuple[str,Tuple[int,int]]]:
    """
    Detect if outputs equal input bbox size, or doubled/halved, etc.
    Returns ('bbox', (factorH, factorW)) or None
    """
    factors = []
    for din, dout in demos:
        bh0, bw0 = _shape_of(dout)
        y0,x0,y1,x1 = _bbox_nonzero(din)
        bh = (y1-y0+1); bw = (x1-x0+1)
        if bh<=0 or bw<=0:
            return None
        # Accept exact match or simple multiples
        fH = bh0 / bh; fW = bw0 / bw
        factors.append((fH,fW))
    # If all near-equal within tolerance
    if not factors: return None
    fH_med = sorted(f[0] for f in factors)[len(factors)//2]
    fW_med = sorted(f[1] for f in factors)[len(factors)//2]
    # Rationalize
    rH = _ratio_rational(fH_med, 8); rW = _ratio_rational(fW_med, 8)
    return ('bbox', (rH.numerator//rH.denominator if rH.denominator==1 else rH, 
                     rW.numerator//rW.denominator if rW.denominator==1 else rW))

def _detect_scaling_pattern(demos_hw: List[Tuple[Tuple[int,int], Tuple[int,int]]], test_input) -> Optional[Tuple[Tuple[int,int], str]]:
    """Detect simple scaling patterns (2x, 3x, 0.5x, etc.)"""
    if not demos_hw:
        return None
    
    # Look for consistent integer or simple fractional scaling
    scale_factors_h = []
    scale_factors_w = []
    
    for (hin, win), (hout, wout) in demos_hw:
        if hin > 0 and win > 0:
            scale_h = hout / hin
            scale_w = wout / win
            scale_factors_h.append(scale_h)
            scale_factors_w.append(scale_w)
    
    if not scale_factors_h:
        return None
        
    # Check if all scaling factors are consistent within tolerance
    avg_scale_h = sum(scale_factors_h) / len(scale_factors_h)
    avg_scale_w = sum(scale_factors_w) / len(scale_factors_w)
    
    tolerance = 0.1
    consistent_h = all(abs(sf - avg_scale_h) < tolerance for sf in scale_factors_h)
    consistent_w = all(abs(sf - avg_scale_w) < tolerance for sf in scale_factors_w)
    
    if consistent_h and consistent_w:
        # Try to find nice integer or simple fraction scaling
        nice_scales_h = [0.5, 1.0, 2.0, 3.0, 4.0, 1/3, 2/3, 1.5, 2.5]
        nice_scales_w = [0.5, 1.0, 2.0, 3.0, 4.0, 1/3, 2/3, 1.5, 2.5]
        
        best_h = min(nice_scales_h, key=lambda x: abs(x - avg_scale_h))
        best_w = min(nice_scales_w, key=lambda x: abs(x - avg_scale_w))
        
        if abs(best_h - avg_scale_h) < tolerance and abs(best_w - avg_scale_w) < tolerance:
            Hin, Win = _shape_of(test_input)
            Hout = int(round(Hin * best_h))
            Wout = int(round(Win * best_w))
            return (Hout, Wout), f"{best_h}x{best_w}"
    
    return None

def _detect_padding_pattern(demos_hw: List[Tuple[Tuple[int,int], Tuple[int,int]]], test_input) -> Optional[Tuple[Tuple[int,int], str]]:
    """Detect padding patterns (add N pixels on each side, center, etc.)"""
    if not demos_hw:
        return None
        
    # Look for consistent padding amounts
    padding_h = []
    padding_w = []
    
    for (hin, win), (hout, wout) in demos_hw:
        if hout >= hin and wout >= win:  # Only consider cases where output is larger
            pad_h = hout - hin
            pad_w = wout - win
            padding_h.append(pad_h)
            padding_w.append(pad_w)
    
    if len(padding_h) < len(demos_hw) // 2:  # Need majority to be padding cases
        return None
        
    # Check consistency
    if len(set(padding_h)) <= 1 and len(set(padding_w)) <= 1:
        pad_h = padding_h[0] if padding_h else 0
        pad_w = padding_w[0] if padding_w else 0
        
        Hin, Win = _shape_of(test_input)
        Hout = Hin + pad_h
        Wout = Win + pad_w
        return (Hout, Wout), f"pad_h{pad_h}_w{pad_w}"
        
    return None

def _detect_rotation_size_change(demos_hw: List[Tuple[Tuple[int,int], Tuple[int,int]]], test_input) -> Optional[Tuple[Tuple[int,int], str]]:
    """Detect size changes due to rotation (H,W) -> (W,H)"""
    if not demos_hw:
        return None
        
    # Check if all demos have (H,W) -> (W,H) transformation (90/270 degree rotation)
    rotation_pattern = True
    for (hin, win), (hout, wout) in demos_hw:
        if not (hin == wout and win == hout):
            rotation_pattern = False
            break
    
    if rotation_pattern:
        Hin, Win = _shape_of(test_input)
        return (Win, Hin), "90deg_rotation"
        
    return None

def get_size_prediction_confidence(demos: List[Tuple[np.ndarray,np.ndarray]], 
                                 test_input: np.ndarray,
                                 predicted_size: Tuple[int,int],
                                 method: str) -> float:
    """Calculate confidence score for a size prediction based on method and consistency"""
    base_confidence = {
        "constant_from_demos": 0.95,
        "affine_int": 0.90, 
        "axis_ratio": 0.85,
        "bbox_": 0.80,
        "scaling_pattern": 0.75,
        "padding_pattern": 0.70,
        "rotation_change": 0.65,
        "fallback_identity": 0.10
    }
    
    # Find base confidence from method
    confidence = 0.50  # Default
    for method_key, base_conf in base_confidence.items():
        if method.startswith(method_key):
            confidence = base_conf
            break
            
    # Adjust confidence based on demo consistency and other factors
    if len(demos) >= 3:  # More demos = higher confidence
        confidence += 0.05
    elif len(demos) == 1:  # Fewer demos = lower confidence
        confidence -= 0.10
        
    # Check size reasonableness
    H_pred, W_pred = predicted_size
    if H_pred <= 0 or W_pred <= 0 or H_pred > 50 or W_pred > 50:
        confidence *= 0.5  # Penalize unreasonable sizes
        
    # Bonus for square outputs (common in ARC)
    if H_pred == W_pred and H_pred in [3, 5, 7, 9, 11]:  # Common ARC sizes
        confidence += 0.02
        
    return min(0.99, max(0.01, confidence))  # Clamp to reasonable range

# Compatibility wrapper for existing code
def predict_size_with_confidence(demos: List[Tuple[np.ndarray,np.ndarray]],
                                test_input: np.ndarray,
                                Hmax:int=30, Wmax:int=30) -> Tuple[Tuple[int,int], float, str]:
    """Wrapper that always returns size prediction with confidence"""
    return predict_size(demos, test_input, Hmax, Wmax, return_confidence=True)
    
def predict_size_simple(demos: List[Tuple[np.ndarray,np.ndarray]],
                       test_input: np.ndarray,
                       Hmax:int=30, Wmax:int=30) -> Tuple[int,int,str]:
    """Wrapper that maintains backward compatibility (no confidence)"""
    return predict_size(demos, test_input, Hmax, Wmax, return_confidence=False)

def predict_size(demos: List[Tuple[np.ndarray,np.ndarray]],
                 test_input: np.ndarray,
                 Hmax:int=30, Wmax:int=30,
                 return_confidence:bool=True) -> Union[Tuple[int,int,str], Tuple[Tuple[int,int], float, str]]:
    """
    Enhanced size prediction with confidence scoring and multiple analysis methods.
    
    Returns:
        If return_confidence=False: (H_out, W_out, reason_string)
        If return_confidence=True: ((H_out, W_out), confidence, reason_string)
    
    Analysis order (with confidence scoring):
      1) Constant output size across demos (conf=0.95)
      2) Small-integer affine map (conf=0.90) 
      3) Axis-wise rational ratios (conf=0.85)
      4) BBox-consistent mapping (conf=0.80)
      5) Scaling pattern detection (conf=0.75)
      6) Padding pattern analysis (conf=0.70)
      7) Rotation size changes (conf=0.65)
      8) Fallback to input size (conf=0.10)
    """
    # Normalize to numpy and collect comprehensive size data
    demos_hw = []
    out_sizes = []
    size_ratios = []
    bbox_ratios = []
    
    for din, dout in demos:
        Hin, Win = _shape_of(din); Hout, Wout = _shape_of(dout)
        demos_hw.append(((Hin,Win),(Hout,Wout)))
        out_sizes.append((Hout,Wout))
        
        # Collect ratio data for pattern analysis
        if Hin > 0 and Win > 0:
            size_ratios.append((Hout/Hin, Wout/Win))
            
            # Bbox ratio analysis
            y0,x0,y1,x1 = _bbox_nonzero(din)
            bbox_h, bbox_w = (y1-y0+1), (x1-x0+1)
            if bbox_h > 0 and bbox_w > 0:
                bbox_ratios.append((Hout/bbox_h, Wout/bbox_w))

    # 1) Constant output size (highest confidence)
    if len(out_sizes)>0 and all(s==out_sizes[0] for s in out_sizes):
        Hf, Wf = out_sizes[0]
        result = (int(max(1,min(Hmax,Hf))), int(max(1,min(Wmax,Wf))))
        confidence = 0.95
        reason = "constant_from_demos"
        if return_confidence:
            return result, confidence, reason
        return result[0], result[1], reason

    # 2) Small-integer affine map (very high confidence)
    fit = _fit_affine_int(demos_hw)
    if fit is not None:
        a1,b1,c1,a2,b2,c2 = fit
        Hin, Win = _shape_of(test_input)
        Hf = a1*Hin + b1*Win + c1
        Wf = a2*Hin + b2*Win + c2
        result = (int(max(1,min(Hmax,Hf))), int(max(1,min(Wmax,Wf))))
        confidence = 0.90
        reason = f"affine_int({a1},{b1},{c1};{a2},{b2},{c2})"
        if return_confidence:
            return result, confidence, reason
        return result[0], result[1], reason

    # 3) Axis-wise rational ratios (high confidence)
    rat = _fit_axiswise_ratio(demos_hw, max_den=12)
    if rat is not None:
        rH, rW = rat
        Hin, Win = _shape_of(test_input)
        Hf = int(round(Hin * float(rH)))
        Wf = int(round(Win * float(rW)))
        result = (int(max(1,min(Hmax,Hf))), int(max(1,min(Wmax,Wf))))
        confidence = 0.85
        reason = f"axis_ratio({rH},{rW})"
        if return_confidence:
            return result, confidence, reason
        return result[0], result[1], reason

    # 4) BBox-consistent mapping (good confidence)
    m = _consistent_bbox_mapping(demos)
    if m is not None:
        mode, (fH,fW) = m
        y0,x0,y1,x1 = _bbox_nonzero(test_input)
        bh = (y1-y0+1); bw = (x1-x0+1)
        # factors may be Fraction or int
        fHf = float(fH) if not isinstance(fH,int) else fH
        fWf = float(fW) if not isinstance(fW,int) else fW
        Hf = int(round(bh * fHf)); Wf = int(round(bw * fWf))
        result = (int(max(1,min(Hmax,Hf))), int(max(1,min(Wmax,Wf))))
        confidence = 0.80
        reason = f"bbox_{fH}x{fW}"
        if return_confidence:
            return result, confidence, reason
        return result[0], result[1], reason

    # 5) Scaling pattern detection (moderate confidence)
    scale_result = _detect_scaling_pattern(demos_hw, test_input)
    if scale_result is not None:
        result, reason_detail = scale_result
        Hf, Wf = result
        result = (int(max(1,min(Hmax,Hf))), int(max(1,min(Wmax,Wf))))
        confidence = 0.75
        reason = f"scaling_pattern({reason_detail})"
        if return_confidence:
            return result, confidence, reason
        return result[0], result[1], reason
    
    # 6) Padding pattern analysis (moderate confidence)
    padding_result = _detect_padding_pattern(demos_hw, test_input)
    if padding_result is not None:
        result, reason_detail = padding_result
        Hf, Wf = result
        result = (int(max(1,min(Hmax,Hf))), int(max(1,min(Wmax,Wf))))
        confidence = 0.70
        reason = f"padding_pattern({reason_detail})"
        if return_confidence:
            return result, confidence, reason
        return result[0], result[1], reason
        
    # 7) Rotation size change detection (lower confidence)
    rotation_result = _detect_rotation_size_change(demos_hw, test_input)
    if rotation_result is not None:
        result, reason_detail = rotation_result
        Hf, Wf = result
        result = (int(max(1,min(Hmax,Hf))), int(max(1,min(Wmax,Wf))))
        confidence = 0.65
        reason = f"rotation_change({reason_detail})"
        if return_confidence:
            return result, confidence, reason
        return result[0], result[1], reason
    
    # 8) Fallback identity (very low confidence)
    Hin, Win = _shape_of(test_input)
    result = (int(Hin), int(Win))
    confidence = 0.10
    reason = "fallback_identity"
    if return_confidence:
        return result, confidence, reason
    return result[0], result[1], reason
