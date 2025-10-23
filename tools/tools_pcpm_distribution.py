# -*- coding: utf-8 -*-
"""
Created on Mon May  5 15:30:03 2025

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

================================ description ==================================

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from Bio import pairwise2

import MDAnalysis as mda
from MDAnalysis.analysis import distances

def map_native2casp(casp_seq: str, native_seq: str) -> List[Tuple[int,int]]:
    if native_seq in casp_seq:                 # 最常见：纯子串
        start = casp_seq.index(native_seq)
        return [(i, i+start) for i in range(len(native_seq))]

    alns = pairwise2.align.globalms(casp_seq, native_seq,
                                    2, -1, -5, -1, one_alignment_only=True)
    q, t = alns[0].seqA, alns[0].seqB   # casp, native
    idx_casp, idx_native, mapping = 0, 0, []
    for qc, tn in zip(q, t):
        if qc !=  '-': idx_casp  += 1
        if tn !=  '-': idx_native += 1
        if qc != '-' and tn != '-':         # match / mismatch
            mapping.append((idx_native-1, idx_casp-1))
    return mapping        # [(native_i, casp_j), ...]


class OnlineMomentMasked:
    """
    逐单元流式累积 1‑4 阶原始矩；finalize 输出:
        0 → mean
        1 → std  (√population variance)
        2 → skew  (E[z³])
        3 → excess kurtosis (E[z⁴] − 3)
    与  scipy.stats.kurtosis(fisher=True, bias=True)  完全一致。
    
    注意： kurt 计算还是错误的，不知道哪里错了，放弃
    
    """
    __slots__ = ("count", "sum1", "sum2", "sum3", "sum4")

    def __init__(self, shape, dtype=torch.float32, device="cpu"):
        self.count = torch.zeros(shape, dtype=torch.float32, device=device)
        self.sum1  = torch.zeros(shape, dtype=dtype,        device=device)
        self.sum2  = torch.zeros_like(self.sum1)
        self.sum3  = torch.zeros_like(self.sum1)
        self.sum4  = torch.zeros_like(self.sum1)
    @torch.no_grad()
    def update(self, x: torch.Tensor, mask: torch.Tensor):
        # mask ∈ {0,1}; x、mask 同形
        self.count += mask
        x = torch.nan_to_num(x)
        self.sum1 += x         * mask
        self.sum2 += (x**2)    * mask
        self.sum3 += (x**3)    * mask
        self.sum4 += (x**4)    * mask
    def finalize(self) -> torch.Tensor:
        n     = self.count                          # element‑wise sample size
        valid = n > 0
        mean = torch.where(valid, self.sum1 / n, torch.nan)
        # population variance
        var  = torch.where(
            valid,
            (self.sum2 / n) - mean**2,
            torch.nan
        )
        std  = torch.sqrt(torch.clamp(var, 1e-12))
        # 3rd, 4th central moment averages
        m3_bar = torch.where(
            valid,
            (self.sum3 / n) - 3*mean*(self.sum2 / n) + 2*mean**3,
            torch.nan
        )
        m4_bar = torch.where(
            valid,
            (self.sum4 / n)
            - 4*mean*(self.sum3 / n)
            + 6*mean**2*(self.sum2 / n)
            - 3*mean**4,
            torch.nan
        )
        skew  = m3_bar / (std**3 + 1e-12)
        kurt  = m4_bar / (var**2 + 1e-12) - 3.0     # Fisher excess
        return torch.stack([mean, std, skew, kurt], dim=0)
    
    
def tri_to_square(vec, L, keep_diag=True, dtype=np.float16):
    k = 0 if keep_diag else 1
    mat = np.zeros((L, L), dtype=dtype)
    idx = np.triu_indices(L, k=k)
    mat[idx] = vec
    mat = mat + mat.T
    if not keep_diag:
        mat[np.diag_indices(L)] = 0
    return mat

def _stack_and_stats(stack: torch.Tensor) -> torch.Tensor:
    """
    Given an M×N×N tensor, return a 3×N×N tensor:
    0 → mean, 1 → std, 2 → skewness  (population formulas).
    """
    mean = stack.mean(dim=0)
    std  = stack.std (dim=0, unbiased=False)
    centered = stack - mean               # broadcast subtraction
    skew = (centered**3).mean(dim=0) / (std**3 + 1e-8)
    return torch.stack([mean, std, skew], dim=0)   # 3 × N × N


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

traintestval_dp = r"/mnt/rna01/zyh/prjs/caspdynamics/data/casp_14_15_labels_traintestval.csv"
traintestval_df = pd.read_csv(traintestval_dp)
traintestval_df.index = traintestval_df['decoy']
table_dp = r"/mnt/rna01/zyh/prjs/caspdynamics/codes/selected_dataset_734.xlsx"
table_df = pd.read_excel(table_dp, index_col=0)

for CASP_ID in casp_ids:
    print(f"{CASP_ID} is processing!")
    casp_seq = table_df.loc[CASP_ID, 'casp_seq']
    native_seq = table_df.loc[CASP_ID, 'native_seq']
    decoys_tdxm_dp = rf"/mnt/dna01/library2/caspdynamics/casp1415_tdxm/casp_14_15_labels_traintestval_{CASP_ID}.pkl"
    output_dp = rf"/mnt/rna01/zyh/prjs/caspdynamics/data/casp_14_15_labels_traintestval_tdxm_pt2/{CASP_ID}_tdxmap.pt"
    # os.makedirs(os.path.dirname(output_dp), exist_ok=True)
    if not os.path.exists(output_dp):
        with open(decoys_tdxm_dp,'rb') as o:
            decoys_tdxm = pickle.load(o)

        bins = np.linspace(0, 1, 33)              # 32 个 bin → idx 0..31
        all_keys = list(range(32)) + ["same_topo", "diff_topo","all"]
        
        # stats_accum[key][map_name] = OnlineMoment(...)
        stats_accum = {k: defaultdict(lambda: None) for k in all_keys}
        native2casp  = map_native2casp(casp_seq, native_seq)   # 只做一次即可
        Lref = len(casp_seq)
        native_casp_idx = torch.tensor([j for _, j in native2casp], dtype=torch.long)
        for k, v in tqdm(decoys_tdxm.items()):
            casp_id   = k.split('_')[0]
            decoy_id  = '_'.join(k.split('_')[1:])          # {casp_id}_{decoy}

            sub_df = traintestval_df[traintestval_df.index == decoy_id]
            tmscore  = sub_df[sub_df['casp_id'] == casp_id]['tmscore'].item()
        
            bin_idx  = int(np.digitize(tmscore, bins[:-1], right=False) - 1)
            topo_key = "same_topo" if tmscore >= 0.5 else "diff_topo"
        
            for map_name in ['cadis', 'omg', 'phi', 'theta']:
                small_vec = v[map_name]                # 1D 或 ndarray
                Lsmall    = v['length']                # decoy 的残基数
                small = tri_to_square(small_vec, Lsmall)
                small_t = torch.as_tensor(small, dtype=torch.float32)
                
                small = tri_to_square(v[map_name], v['length'])      # (Ldecoy,Ldecoy)
                big   = torch.full((Lref, Lref), torch.nan, dtype=torch.float32)
                if Lsmall == Lref:
                    small_idx = torch.arange(Lref)               # 0..L-1
                    big_idx   = small_idx
                elif Lsmall == len(native_seq):
                    small_idx = torch.arange(Lsmall)             # decoy 索引
                    big_idx   = native_casp_idx.clone()
                    
                sub_small = small_t[small_idx.unsqueeze(1), small_idx.unsqueeze(0)]
                # big[np.ix_(idx, idx)] = small                        # 写入子矩阵
                big[big_idx.unsqueeze(1), big_idx.unsqueeze(0)] = sub_small    # (k,1) × (1,k)
                mask = ~torch.isnan(big)
                for key in (bin_idx, topo_key, "all"):
                    if stats_accum[key][map_name] is None:
                        stats_accum[key][map_name] = OnlineMomentMasked(big.shape,
                                                                        dtype=big.dtype,
                                                                        device=big.device)
                    stats_accum[key][map_name].update(big, mask.float())

        stats_dict = {}          # stats_dict[key][map_name] = 3×N×N
        for key, d in stats_accum.items():
            stats_dict[key] = {}
            for map_name, acc in d.items():
                if acc is None:          # 该 bin 可能一个样本都没有
                    continue
                stats_dict[key][map_name] = acc.finalize()
        stats_dict['casp_seq'] = casp_seq
        stats_dict['native_seq'] = native_seq
        
        torch.save(stats_dict, output_dp)



