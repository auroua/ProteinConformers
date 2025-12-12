# -*- coding: utf-8 -*-
"""
Created on Mon May  5 15:30:03 2025

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

================================ description ==================================
update from tools_pre_pcpm_to_pcpm.py ：
1. update OnlineMoment，use circular computation, and only consider interaction within 20A distance

=================================== input =====================================

=================================== output ====================================

================================= parameters ==================================

=================================== example ===================================

=================================== warning ===================================
"""

from __future__ import annotations

import pickle
import argparse
import os
import glob
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
from collections import defaultdict
import sys
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

import MDAnalysis as mda
from MDAnalysis.analysis import distances

class LinearOnlineMoment:
    def __init__(self, shape, device='cpu', dtype=torch.float32):
        self.count = 0
        self.sum1 = torch.zeros(shape, device=device, dtype=dtype)
        self.sum2 = torch.zeros_like(self.sum1)
        self.sum3 = torch.zeros_like(self.sum1)
        self.sum4 = torch.zeros_like(self.sum1)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if x.ndim == 3:  
            x = x.squeeze(0)
        self.count += 1
        self.sum1 += x
        self.sum2 += x * x
        self.sum3 += x * x * x
        self.sum4 += x * x * x * x

    def finalize(self) -> torch.Tensor:
        n = float(self.count)
        mean = self.sum1 / n
        m2 = self.sum2 / n
        m3 = self.sum3 / n
        m4 = self.sum4 / n

        var = torch.clamp(m2 - mean ** 2, min=1e-12)
        std = torch.sqrt(var)

        skew = torch.clamp((m3 - 3 * mean * var - mean ** 3) / (std ** 3 + 1e-12), min=-1e6, max=1e6)
        kurt = torch.clamp(
            (m4 - 4 * mean * m3 + 6 * mean ** 2 * m2 - 3 * mean ** 4) / (std ** 4 + 1e-12),
            min=-1e6, max=1e6
        )
        return torch.stack([mean, std, skew, kurt], dim=0)  # 4×N×N

class CircularOnlineMoment:
    """
    circular (rad) ：
    -  ∑cosθ, ∑sinθ, ∑cos2θ, ∑sin2θ
    - output：mean(μ), std_c(√(-2ln R̄)), skew(γ1), kurt(γ2)
    """
    def __init__(self, shape, device='cpu', dtype=torch.float32, eps=1e-12):
        self.count = 0
        self.sum_c1 = torch.zeros(shape, device=device, dtype=dtype)
        self.sum_s1 = torch.zeros_like(self.sum_c1)
        self.sum_c2 = torch.zeros_like(self.sum_c1)
        self.sum_s2 = torch.zeros_like(self.sum_c1)
        self.eps = torch.as_tensor(eps, device=device, dtype=dtype)
    @torch.no_grad()
        """x: Angle (radian), tensor shape N×N or 1×N×N"""
        if x.ndim == 3:
            x = x.squeeze(0)
        self.count += 1
        # First-order and second-order trigonometric moments
        self.sum_c1 += torch.cos(x)
        self.sum_s1 += torch.sin(x)
        self.sum_c2 += torch.cos(2.0 * x)
        self.sum_s2 += torch.sin(2.0 * x)
    def finalize(self) -> torch.Tensor:
        n = torch.as_tensor(float(self.count), device=self.sum_c1.device, dtype=self.sum_c1.dtype)
        a1 = self.sum_c1 / (n + self.eps)
        b1 = self.sum_s1 / (n + self.eps)
        a2 = self.sum_c2 / (n + self.eps)
        b2 = self.sum_s2 / (n + self.eps)
        # Mean direction μ, combined vector length R̄
        mu = torch.atan2(b1, a1)                              # [-π, π)
        Rbar = torch.clamp(torch.sqrt(a1 * a1 + b1 * b1), 0.0, 1.0)
        # Standard deviation of a circle: √(-2 ln R̄), when R̄≈1/0, perform clipping.
        std_c = torch.sqrt(torch.clamp(-2.0 * torch.log(torch.clamp(Rbar, min=1e-8)), min=0.0))
        # Centralized second-order trigonometric matrix
        # κ2 = E[cos 2(θ-μ)]，β2 = E[sin 2(θ-μ)]
        cos2mu = torch.cos(2.0 * mu)
        sin2mu = torch.sin(2.0 * mu)
        kappa2 = a2 * cos2mu + b2 * sin2mu
        beta2  = b2 * cos2mu - a2 * sin2mu
        one_minus_R = torch.clamp(1.0 - Rbar, min=1e-8)
        skew = beta2 / (torch.pow(one_minus_R, 1.5) + self.eps)                    # γ1
        kurt = (kappa2 - Rbar ** 4) / (one_minus_R ** 2 + self.eps)                # γ2
        return torch.stack([mu, std_c, skew, kurt], dim=0)  # 4×N×N


def _pad_square(t: torch.Tensor, target_L: int, pad_value: float):
    L = t.shape[0]
    if L == target_L:
        return t
    out = torch.full((target_L, target_L), pad_value, dtype=t.dtype, device=t.device)
    out[:L, :L] = t
    return out

class MaskedLinearOnlineMoment:
    def __init__(self, shape, device='cpu', dtype=torch.float32, eps=1e-12):
        self.count = torch.zeros(shape, device=device, dtype=torch.int32)
        self.sum1  = torch.zeros(shape, device=device, dtype=dtype)
        self.sum2  = torch.zeros_like(self.sum1)
        self.sum3  = torch.zeros_like(self.sum1)
        self.sum4  = torch.zeros_like(self.sum1)
        self.eps   = torch.as_tensor(eps, device=device, dtype=dtype)
        self.default_cutoff = 20.0

    def _grow(self, new_L: int):
        """Expand the buffer to new_L × new_L (only if it increases)."""
        old_L = self.count.shape[0]
        if new_L <= old_L:
            return
        def grow_like(t, fill=0):
            out = torch.full((new_L, new_L), fill, dtype=t.dtype, device=t.device)
            out[:old_L, :old_L] = t
            return out
        self.count = grow_like(self.count, 0)
        self.sum1  = grow_like(self.sum1,  0.0)
        self.sum2  = grow_like(self.sum2,  0.0)
        self.sum3  = grow_like(self.sum3,  0.0)
        self.sum4  = grow_like(self.sum4,  0.0)

    @torch.no_grad()
    def update(self, x: torch.Tensor, cadis: torch.Tensor, cutoff: float = 20.0):
        if x.ndim == 3: x = x.squeeze(0)
        if cadis.ndim == 3: cadis = cadis.squeeze(0)
        assert x.shape == cadis.shape, f"x and cadis shape mismatch: {x.shape} vs {cadis.shape}"

        L = x.shape[0]
        # Expand to a larger L
        self._grow(L)
        target_L = self.count.shape[0]

        # Fill in the current statistical shape
        if L != target_L:
            x     = _pad_square(x,     target_L, 0.0)
            cadis = _pad_square(cadis, target_L, float('inf'))

        mask = (cadis <= cutoff) & torch.isfinite(x)
        xm = torch.where(mask, x, torch.zeros_like(x))

        self.count += mask.to(self.count.dtype)
        self.sum1  += xm
        self.sum2  += xm * xm
        self.sum3  += xm * xm * xm
        self.sum4  += xm * xm * xm * xm

    def finalize(self) -> torch.Tensor:
        n = self.count.to(self.sum1.dtype).clamp(min=1.0)
        mean = self.sum1 / n
        m2   = self.sum2 / n
        m3   = self.sum3 / n
        m4   = self.sum4 / n
        var  = torch.clamp(m2 - mean ** 2, min=self.eps.item())
        std  = torch.sqrt(var)
        skew = torch.clamp((m3 - 3 * mean * var - mean ** 3) / (std ** 3 + self.eps), min=-1e6, max=1e6)
        kurt = torch.clamp((m4 - 4 * mean * m3 + 6 * mean ** 2 * m2 - 3 * mean ** 4) / (std ** 4 + self.eps),
                           min=-1e6, max=1e6)
        zero_mask = (self.count == 0)
        mean = torch.where(zero_mask, torch.zeros_like(mean), mean)
        std  = torch.where(zero_mask, torch.zeros_like(std),  std)
        skew = torch.where(zero_mask, torch.zeros_like(skew), skew)
        kurt = torch.where(zero_mask, torch.zeros_like(kurt), kurt)
        return torch.stack([mean, std, skew, kurt], dim=0)

class MaskedCircularOnlineMoment:
    def __init__(self, shape, device='cpu', dtype=torch.float32, eps=1e-12):
        self.count  = torch.zeros(shape, device=device, dtype=torch.int32)
        self.sum_c1 = torch.zeros(shape, device=device, dtype=dtype)
        self.sum_s1 = torch.zeros_like(self.sum_c1)
        self.sum_c2 = torch.zeros_like(self.sum_c1)
        self.sum_s2 = torch.zeros_like(self.sum_c1)
        self.eps    = torch.as_tensor(eps, device=device, dtype=dtype)
        self.default_cutoff = 20.0

    def _grow(self, new_L: int):
        old_L = self.count.shape[0]
        if new_L <= old_L:
            return
        def grow_like(t, fill=0.0):
            out = torch.full((new_L, new_L), fill, dtype=t.dtype, device=t.device)
            out[:old_L, :old_L] = t
            return out
        self.count  = grow_like(self.count, 0)
        self.sum_c1 = grow_like(self.sum_c1, 0.0)
        self.sum_s1 = grow_like(self.sum_s1, 0.0)
        self.sum_c2 = grow_like(self.sum_c2, 0.0)
        self.sum_s2 = grow_like(self.sum_s2, 0.0)

    @torch.no_grad()
    def update(self, x: torch.Tensor, cadis: torch.Tensor, cutoff: float = 20.0):
        if x.ndim == 3: x = x.squeeze(0)
        if cadis.ndim == 3: cadis = cadis.squeeze(0)
        assert x.shape == cadis.shape, f"x and cadis shape mismatch: {x.shape} vs {cadis.shape}"

        L = x.shape[0]
        self._grow(L)
        target_L = self.count.shape[0]

        if L != target_L:
            x     = _pad_square(x,     target_L, 0.0)           # The value doesn't matter, it will be masked by Cadis anyway.
            cadis = _pad_square(cadis, target_L, float('inf'))  # Fill in the blank with an integer to ensure it is not included.

        mask = (cadis <= cutoff) & torch.isfinite(x)
        cx1 = torch.where(mask, torch.cos(x), torch.zeros_like(x))
        sx1 = torch.where(mask, torch.sin(x), torch.zeros_like(x))
        cx2 = torch.where(mask, torch.cos(2.0 * x), torch.zeros_like(x))
        sx2 = torch.where(mask, torch.sin(2.0 * x), torch.zeros_like(x))

        self.count  += mask.to(self.count.dtype)
        self.sum_c1 += cx1
        self.sum_s1 += sx1
        self.sum_c2 += cx2
        self.sum_s2 += sx2

    def finalize(self) -> torch.Tensor:
        n = self.count.to(self.sum_c1.dtype).clamp(min=1.0)
        a1 = self.sum_c1 / (n + self.eps)
        b1 = self.sum_s1 / (n + self.eps)
        a2 = self.sum_c2 / (n + self.eps)
        b2 = self.sum_s2 / (n + self.eps)
        mu   = torch.atan2(b1, a1)
        Rbar = torch.clamp(torch.sqrt(a1*a1 + b1*b1), 0.0, 1.0)
        std_c = torch.sqrt(torch.clamp(-2.0 * torch.log(torch.clamp(Rbar, min=1e-8)), min=0.0))
        cos2mu = torch.cos(2.0 * mu)
        sin2mu = torch.sin(2.0 * mu)
        kappa2 = a2 * cos2mu + b2 * sin2mu
        beta2  = b2 * cos2mu - a2 * sin2mu
        one_minus_R = torch.clamp(1.0 - Rbar, min=1e-8)
        skew = beta2 / (torch.pow(one_minus_R, 1.5) + self.eps)
        kurt = (kappa2 - Rbar ** 4) / (one_minus_R ** 2 + self.eps)
        zero_mask = (self.count == 0)
        mu    = torch.where(zero_mask, torch.zeros_like(mu),    mu)
        std_c = torch.where(zero_mask, torch.zeros_like(std_c), std_c)
        skew  = torch.where(zero_mask, torch.zeros_like(skew),  skew)
        kurt  = torch.where(zero_mask, torch.zeros_like(kurt),  kurt)
        return torch.stack([mu, std_c, skew, kurt], dim=0)


def get_moment_accumulator(map_name,
                           shape,
                           device='cpu',
                           dtype=torch.float32,
                           *,
                           cutoff: float = 20.0):
    """
    Returns an element-wise accumulator with a "20Å CA distance mask".
    Note: When calling `update`, you need to pass in (x, cadis), for example:
    `acc = get_moment_accumulator('phi', (L, L), device=x.device, dtype=x.dtype, cutoff=20.0)`
    `acc.update(phi_map, cadis_map)` # Only accumulates (i,j) values ​​where cadis <= 20Å    
    """
    ANGLE_MAPS = {'omg', 'phi', 'theta'}
    if map_name in ANGLE_MAPS:
        acc = MaskedCircularOnlineMoment(shape, device=device, dtype=dtype)
    else:
        acc = MaskedLinearOnlineMoment(shape, device=device, dtype=dtype)
    # Bind a default cutoff to the instance to facilitate reuse when it is not passed externally.
    acc.default_cutoff = float(cutoff)
    # Wrap it in a layer: This allows you to use either `acc.update(x, cadis)` or `acc.update(x, cadis, cutoff=...)`.
    orig_update = acc.update
    def _update_with_default_cutoff(x, cadis, cutoff=None):
        c = acc.default_cutoff if (cutoff is None) else float(cutoff)
        return orig_update(x, cadis, cutoff=c)
    acc.update = _update_with_default_cutoff
    return acc

def tri_to_square(vec, L, keep_diag=True, dtype=np.float16):
    k = 0 if keep_diag else 1
    mat = np.zeros((L, L), dtype=dtype)
    idx = np.triu_indices(L, k=k)
    mat[idx] = vec
    mat = mat + mat.T
    if not keep_diag:
        mat[np.diag_indices(L)] = 0
    return mat

if __name__ == '__main__':
    casp_ids = ['T1024',
     'T1025',
     'T1026',
     'T1027',
     'T1028',
     'T1029',
     'T1030',
     'T1031',
     'T1032',
     'T1033',
     'T1034',
     'T1035',
     'T1036s1',
     'T1037',
     'T1038',
     'T1039',
     'T1040',
     'T1041',
     'T1042',
     'T1043',
     'T1045s1',
     'T1045s2',
     'T1046s1',
     'T1046s2',
     'T1047s1',
     'T1047s2',
     'T1048',
     'T1049',
     'T1050',
     'T1052',
     'T1053',
     'T1054',
     'T1055',
     'T1056',
     'T1057',
     'T1058',
     'T1060s2',
     'T1060s3',
     'T1061',
     'T1062',
     'T1064',
     'T1065s1',
     'T1065s2',
     'T1067',
     'T1068',
     'T1070',
     'T1072s1',
     'T1073',
     'T1074',
     'T1076',
     'T1078',
     'T1079',
     'T1080',
     'T1082',
     'T1083',
     'T1084',
     'T1087',
     'T1088',
     'T1089',
     'T1090',
     'T1091',
     'T1092',
     'T1093',
     'T1094',
     'T1095',
     'T1096',
     'T1098',
     'T1099',
     'T1100',
     'T1101',
     'T1104',
     'T1106s1',
     'T1106s2',
     'T1109',
     'T1119',
     'T1123',
     'T1124',
     'T1129s2',
     'T1133',
     'T1137s7',
     'T1137s8',
     'T1137s9',
     'T1139',
     'T1150',
     'T1180',
     'T1188',
     'T1194']

    # method_name = 'alphaflow_md'
    method_name = sys.argv[1]

    table_dp = r"/mnt/rna01/zyh/prjs/caspdynamics/codes/selected_dataset_734.xlsx"
    traintestval_dir = rf"/mnt/dna01/library2/caspdynamics/esmflow_decoys/{method_name}_distilled"
    table_df = pd.read_excel(table_dp, index_col=0)

    for CASP_ID in casp_ids:
        casp_seq = table_df.loc[CASP_ID, 'casp_seq']
        native_seq = table_df.loc[CASP_ID, 'native_seq']
        csv_dp = os.path.join(traintestval_dir,f"{CASP_ID}_decoys.csv")
        if os.path.exists(csv_dp):
            traintestval_df = pd.read_csv(csv_dp)
            traintestval_df.index = [os.path.basename(_) for _ in traintestval_df['decoy']]
        else:
            continue
        print(f"{CASP_ID} is processing!")
        decoys_tdxm_dp = rf"/mnt/dna01/library2/caspdynamics/{method_name}_distilled_new_tdxm/{method_name}_distilled_new_{CASP_ID}_tdxm_{CASP_ID}.pkl"
        output_dp = rf"/mnt/rna01/zyh/prjs/caspdynamics/data/{method_name}_new_tdxm_distribution/{method_name}_{CASP_ID}_tdxmap.pkl"
        os.makedirs(os.path.dirname(output_dp), exist_ok=True)
        
        if os.path.exists(decoys_tdxm_dp):
            with open(decoys_tdxm_dp,'rb') as o:
                decoys_tdxm = pickle.load(o)
        else:
            print(f"cannot find {decoys_tdxm_dp}, skip!")
            continue
        
        # ─────────────────────────────────────────────
        # 2) Main loop: Iterate through decoy while accumulating online.
        # ─────────────────────────────────────────────
        bins = np.linspace(0, 1, 33)              # 32 个 bin → idx 0..31
        all_keys = list(range(32)) + ["same_topo", "diff_topo", "all"]
        
        # stats_accum[key][map_name] = OnlineMoment(...)
        stats_accum = {k: defaultdict(lambda: None) for k in all_keys}
        
        for k, v in tqdm(decoys_tdxm.items()):
            casp_id   = k.split('_')[0]
            decoy_id  = '_'.join(k.split('_')[1:])          # {casp_id}_{decoy}
        
            sub_df = traintestval_df[traintestval_df.index == decoy_id]
            tmscore  = sub_df[sub_df['casp_id'] == casp_id]['tmscore'].item()
        
            bin_idx  = int(np.digitize(tmscore, bins[:-1], right=False) - 1)
            topo_key = "same_topo" if tmscore >= 0.5 else "diff_topo"
        
            try: 
                # Sometimes the residue numbers are inconsistent; for example, the native residue number may be less than the decoys residue number.
                # —— Four types of contact maps ——————————————————————————
                # First, obtain the Cadis matrix (same as before).
                cadis_sq = tri_to_square(v['cadis'], v['length'])
                cadis_sq = torch.as_tensor(cadis_sq, dtype=torch.float32, device='cpu') 
                for map_name in ['cadis', 'omg', 'phi', 'theta']:
                    map_ = tri_to_square(v[map_name], v['length'])       # → N×N torch.Tensor/ndarray
                    if not isinstance(map_, torch.Tensor):
                        map_ = torch.as_tensor(map_, dtype=torch.float32)
            
                    # Initialize OnlineMoment (only on the first occurrence)
                    for key in (bin_idx, topo_key, "all"):
                        if stats_accum[key][map_name] is None:
                            stats_accum[key][map_name] = get_moment_accumulator(
                                map_name, map_.shape, device=map_.device, dtype=map_.dtype, cutoff=20.0
                            )
                        # Change: Pass cadis along with the update; internally, it will automatically only count (i,j) values ​​≤20Å.
                        stats_accum[key][map_name].update(map_, cadis_sq)
            except Exception as e:
                print(f"{k} has issue, skip!")
                traceback.print_exc()
        
        stats_dict = {}          # stats_dict[key][map_name] = 3×N×N
        for key, d in stats_accum.items():
            stats_dict[key] = {}
            for map_name, acc in d.items():
                if acc is None:
                    continue
                stats_dict[key][map_name] = acc.finalize()
        
        torch.save(stats_dict, output_dp)



