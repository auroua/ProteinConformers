# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:12:11 2025

@author: Yihang (Ethan) Zhou


Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

================================ description ==================================
calculate the pcps from previously calculated pcpm.
although here the vector dim is 4, but we only use 3 of the 4 dimension.
=================================== input =====================================

=================================== output ====================================

================================= parameters ==================================

=================================== example ===================================

=================================== warning ===================================

"""
import os
import sys
import traceback
import pickle
import pandas as pd
import numpy as np
from Bio import pairwise2
from tqdm import tqdm
import torch
import torch.nn.functional as F

def tri_to_square(vec, L, keep_diag=True, dtype=np.float16):
    k = 0 if keep_diag else 1
    mat = np.zeros((L, L), dtype=dtype)
    idx = np.triu_indices(L, k=k)
    mat[idx] = vec
    mat = mat + mat.T
    if not keep_diag:
        mat[np.diag_indices(L)] = 0
    return mat

def align_and_crop(matrix1, matrix2, seq1, seq2):
    """
    Align seq1 and seq2, then crop BOTH matrix1 and matrix2 to the positions where
    both sequences have non-gap residues. Return (cropped_matrix1, cropped_matrix2).
    兼容 torch / numpy；索引只打在维度 1/2 上。
    """
    # 形状检查：都应是 (4, L, L)
    assert matrix1.shape[0] == 4 and matrix1.shape[1] == matrix1.shape[2], f"matrix1 bad shape {tuple(matrix1.shape)}"
    assert matrix2.shape[0] == 4 and matrix2.shape[1] == matrix2.shape[2], f"matrix2 bad shape {tuple(matrix2.shape)}"
    # 用矩阵边长限制序列，避免 len(seq) 与矩阵边长不一致
    N1 = int(matrix1.shape[1])
    N2 = int(matrix2.shape[1])
    seq1_use = seq1[:N1]
    seq2_use = seq2[:N2]
    # 全局比对
    alignment = pairwise2.align.globalxx(seq1_use, seq2_use)[0]
    a1, a2 = alignment.seqA, alignment.seqB
    # 收集“双方都不是 gap”的列。分别记录落在 seq1_use 和 seq2_use 的原始索引
    idx1, idx2 = [], []
    i1 = i2 = 0
    for c1, c2 in zip(a1, a2):
        if c1 != '-' and c2 != '-':
            idx1.append(i1)
            idx2.append(i2)
        if c1 != '-':
            i1 += 1
        if c2 != '-':
            i2 += 1
    if len(idx1) == 0:
        raise ValueError("No aligned residue positions between seq1 and seq2.")
    # 边界裁剪，避免越界
    if isinstance(matrix1, torch.Tensor):
        t_idx1 = torch.tensor(idx1, dtype=torch.long, device=matrix1.device)
        t_idx2 = torch.tensor(idx2, dtype=torch.long, device=matrix2.device)
        t_idx1 = t_idx1[(t_idx1 >= 0) & (t_idx1 < N1)]
        t_idx2 = t_idx2[(t_idx2 >= 0) & (t_idx2 < N2)]
        if t_idx1.numel() == 0 or t_idx2.numel() == 0:
            raise ValueError("Aligned indices empty after bounds check.")
        # 只在维度 1/2 上索引，避免高级索引歧义
        cropped_matrix1 = matrix1.index_select(1, t_idx1).index_select(2, t_idx1)
        cropped_matrix2 = matrix2.index_select(1, t_idx2).index_select(2, t_idx2)
    else:
        # numpy 情况
        n_idx1 = np.asarray(idx1, dtype=np.int64)
        n_idx2 = np.asarray(idx2, dtype=np.int64)
        n_idx1 = n_idx1[(n_idx1 >= 0) & (n_idx1 < N1)]
        n_idx2 = n_idx2[(n_idx2 >= 0) & (n_idx2 < N2)]
        if n_idx1.size == 0 or n_idx2.size == 0:
            raise ValueError("Aligned indices empty after bounds check.")
        cropped_matrix1 = matrix1[:, n_idx1][:, :, n_idx1]
        cropped_matrix2 = matrix2[:, n_idx2][:, :, n_idx2]
    # 两边都会变成 (4, K, K)，K 为对齐后共同的非 gap 列数
    return cropped_matrix1, cropped_matrix2



def cosine_similarity_across_channels_torch(matrix1: torch.Tensor, matrix2: torch.Tensor) -> torch.Tensor:
    """
    计算两个形状为 (4, N, N) 的张量在每个 (i, j) 点上的通道维度余弦相似度，并对 N×N 平均。

    :param matrix1: torch.Tensor of shape (4, N, N)
    :param matrix2: torch.Tensor of shape (4, N, N)
    :return: scalar tensor, 平均余弦相似度
    """
    assert matrix1.shape == matrix2.shape, "两个矩阵形状必须一致"
    assert matrix1.shape[0] == 4, "矩阵的第一个维度应为 4"

    # 展开为 (4, N*N)，然后转置为 (N*N, 4)
    vec1 = matrix1.view(4, -1).T  # shape: (N*N, 4)
    vec2 = matrix2.view(4, -1).T  # shape: (N*N, 4)

    # F.cosine_similarity 会按最后一个维度计算余弦相似度
    sim = F.cosine_similarity(vec1, vec2, dim=-1)  # shape: (N*N,)
    return sim.mean()

def cosine_similarity_across_first_three_channels(
    matrix1: torch.Tensor,
    matrix2: torch.Tensor
) -> torch.Tensor:
    """
    计算两个形状为 (4, N, N) 的张量在前三个通道维度上的余弦相似度，并对 N×N 点平均。

    :param matrix1: torch.Tensor of shape (4, N, N)
    :param matrix2: torch.Tensor of shape (4, N, N)
    :return: scalar tensor，前三通道平均余弦相似度
    """
    # 检查输入维度
    assert matrix1.shape == matrix2.shape, "两个矩阵形状必须一致"
    assert matrix1.dim() == 3 and matrix1.shape[0] == 4, "矩阵应为 (4, N, N)"

    # 仅取前三个通道
    m1 = matrix1[:3]  # shape: (3, N, N)
    m2 = matrix2[:3]  # shape: (3, N, N)

    # 展开为 (3, N*N)，然后转置为 (N*N, 3)
    vec1 = m1.view(3, -1).T  # shape: (N*N, 3)
    vec2 = m2.view(3, -1).T  # shape: (N*N, 3)

    # 计算每个点在通道维度上的余弦相似度
    sim = F.cosine_similarity(vec1, vec2, dim=-1)  # shape: (N*N,)

    # 返回所有点的平均相似度
    return sim.mean()


def rbf_mahalanobis_similarity_first_three_channels(
    matrix1: torch.Tensor,
    matrix2: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    使用“马氏距离 + RBF”计算两个 (4, N, N) 张量在前三个通道上的相似度：
        d_M(u, v) = sqrt((u - v)^T Σ^{-1} (u - v))
        s(u, v)   = exp( - d_M(u, v)^2 / 2 ) = exp( - ((u - v)^T Σ^{-1} (u - v)) / 2 )
    其中 Σ 用两张图在前三通道拼接后的经验协方差（加 eps*I 做稳定化）。
    返回：对所有 (i,j) 的相似度取均值的 scalar tensor。
    """
    assert matrix1.shape == matrix2.shape and matrix1.dim() == 3 and matrix1.shape[0] == 4, "矩阵应为 (4, N, N)"
    # 仅取前三个通道，并展平为 (N*N, 3)
    m1 = matrix1[:3].reshape(3, -1).T  # (M, 3), M = N*N
    m2 = matrix2[:3].reshape(3, -1).T  # (M, 3)
    # 估计协方差 Σ（对两者拼接后做经验协方差），并加 eps*I 保证可逆
    X = torch.cat([m1, m2], dim=0)                    # (2M, 3)
    Xc = X - X.mean(dim=0, keepdim=True)              # 去均值
    cov = Xc.t().matmul(Xc) / max(Xc.shape[0] - 1, 1) # (3, 3)
    cov = cov + eps * torch.eye(3, dtype=X.dtype, device=X.device)
    inv_cov = torch.inverse(cov)
    # 计算每个像素向量差的二次型 (u-v)^T Σ^{-1} (u-v)
    diff = m1 - m2                                    # (M, 3)
    quad = (diff @ inv_cov) * diff                    # (M, 3)
    dist2 = quad.sum(dim=1)                           # (M,)
    # RBF 相似度，并对所有位置取平均
    sim = torch.exp(-0.5 * dist2)                     # (M,)
    return sim.mean()                                 # scalar tensor

table_dp = r"/mnt/rna01/zyh/prjs/proteinconformers/codes/selected_dataset_734.xlsx"
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

table_df = pd.read_excel(table_dp, index_col=0)
all_keys = list(range(32)) + ["same_topo", "diff_topo","all"]
maps = ['cadis','omg','phi','theta']
if True:
    method_name = 'alphaflow_md'
    results = {}
    # although here the vector dim is 4, but we only use 3 of the 4 dimension.
    output_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/results/{method_name}_pcps_rbf_sim.4.pkl"
    for casp_id in tqdm(casp_ids):
        # pcpm_method_dp means the pcpm of alphaflow{method}
        # pcpm_proteinconformers_dp means the pcpm of proteinconformers{reference}
        # this script is to calculate the divergence between the two pcpm distribution
        pcpm_method_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/data/{method_name}_new_pcpm_distribution/{method_name}_{casp_id}_pcpmap.pkl"
        pcpm_proteinconformers_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/data/casp1415_new_pcpm_distribution/casp1415_{casp_id}_pcpmap.pkl"
        
        sequence_method = table_df.loc[casp_id]['native_seq']
        sequence_proteinconformers = table_df.loc[casp_id]['casp_seq']
        
        if os.path.exists(pcpm_method_dp):
            pcpm_method = torch.load(pcpm_method_dp, weights_only=True)
            pcpm_proteinconformers = torch.load(pcpm_proteinconformers_dp,  weights_only=True)
            results[casp_id] = {}
            for key in all_keys:
                # matrix_method = pcpm_method[key]
                matrix_method = pcpm_method['all']
                matrix_proteinconformers = pcpm_proteinconformers[key]
                if matrix_method and matrix_proteinconformers:
                    results[casp_id][key] = []
                    if len(matrix_proteinconformers.keys()) == 4:
                        for m in maps:
                            try:
                                matrix_proteinconformers_aligned = align_and_crop( matrix_proteinconformers[m],matrix_method[m],
                                                                           sequence_proteinconformers, sequence_proteinconformers, )
                                matrix_proteinconformers_aligned, matrix_method_aligned = align_and_crop(
                                        matrix_proteinconformers[m], matrix_method[m], sequence_proteinconformers, sequence_proteinconformers)
                            except:
                                try:
                                    matrix_proteinconformers_aligned, matrix_method_aligned = align_and_crop(
                                        matrix_proteinconformers[m], matrix_method[m], sequence_proteinconformers, sequence_method)
                                except:
                                    traceback.print_exc()
                                    sys.exit(f"{pcpm_method_dp} {m}")
                            # cos_sim = cosine_similarity_across_first_three_channels(matrix_proteinconformers_aligned,matrix_method[m])
                            # print(f"matrix_proteinconformers_aligned: {matrix_proteinconformers_aligned.shape}")
                            # print(f" matrix_method[m]: {matrix_method[m].shape}")
                            cos_sim = rbf_mahalanobis_similarity_first_three_channels(matrix_proteinconformers_aligned, matrix_method_aligned)
                            results[casp_id][key].append(cos_sim.item())
    with open(output_dp,'wb') as o:
        pickle.dump(results,o)


if True:
    method_name = 'alphaflow_pdb'
    results = {}
    output_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/results/{method_name}_pcps_rbf_sim.4.pkl"
    for casp_id in tqdm(casp_ids):
        pcpm_method_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/data/{method_name}_new_pcpm_distribution/{method_name}_{casp_id}_pcpmap.pkl"
        pcpm_proteinconformers_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/data/casp1415_new_pcpm_distribution/casp1415_{casp_id}_pcpmap.pkl"
        
        sequence_method = table_df.loc[casp_id]['native_seq']
        sequence_proteinconformers = table_df.loc[casp_id]['casp_seq']
        if os.path.exists(pcpm_method_dp):
            pcpm_method = torch.load(pcpm_method_dp, weights_only=True)
            pcpm_proteinconformers = torch.load(pcpm_proteinconformers_dp,  weights_only=True)
            results[casp_id] = {}
            for key in all_keys:
                # matrix_method = pcpm_method[key]
                matrix_method = pcpm_method['all']
                matrix_proteinconformers = pcpm_proteinconformers[key]
                if matrix_method and matrix_proteinconformers:
                    results[casp_id][key] = []
                    if len(matrix_proteinconformers.keys()) == 4:
                        for m in maps:
                            try:
                                matrix_proteinconformers_aligned = align_and_crop( matrix_proteinconformers[m],matrix_method[m],
                                                                           sequence_proteinconformers, sequence_proteinconformers, )
                                matrix_proteinconformers_aligned, matrix_method_aligned = align_and_crop(
                                        matrix_proteinconformers[m], matrix_method[m], sequence_proteinconformers, sequence_proteinconformers)
                            except:
                                try:
                                    matrix_proteinconformers_aligned, matrix_method_aligned = align_and_crop(
                                        matrix_proteinconformers[m], matrix_method[m], sequence_proteinconformers, sequence_method)
                                except:
                                    traceback.print_exc()
                                    sys.exit(f"{pcpm_method_dp} {m}")
                            # cos_sim = cosine_similarity_across_first_three_channels(matrix_proteinconformers_aligned,matrix_method[m])
                            # print(f"matrix_proteinconformers_aligned: {matrix_proteinconformers_aligned.shape}")
                            # print(f" matrix_method[m]: {matrix_method[m].shape}")
                            cos_sim = rbf_mahalanobis_similarity_first_three_channels(matrix_proteinconformers_aligned, matrix_method_aligned)
                            results[casp_id][key].append(cos_sim.item())
    with open(output_dp,'wb') as o:
        pickle.dump(results,o)

if True:
    method_name = 'esmflow_md'
    results = {}
    output_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/results/{method_name}_pcps_rbf_sim.4.pkl"
    for casp_id in tqdm(casp_ids):
        pcpm_method_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/data/{method_name}_new_pcpm_distribution/{method_name}_{casp_id}_pcpmap.pkl"
        pcpm_proteinconformers_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/data/casp1415_new_pcpm_distribution/casp1415_{casp_id}_pcpmap.pkl"
        
        sequence_method = table_df.loc[casp_id]['native_seq']
        sequence_proteinconformers = table_df.loc[casp_id]['casp_seq']
        if os.path.exists(pcpm_method_dp):
            pcpm_method = torch.load(pcpm_method_dp, weights_only=True)
            pcpm_proteinconformers = torch.load(pcpm_proteinconformers_dp,  weights_only=True)
            results[casp_id] = {}
            for key in all_keys:
                # matrix_method = pcpm_method[key]
                matrix_method = pcpm_method['all']
                matrix_proteinconformers = pcpm_proteinconformers[key]
                if matrix_method and matrix_proteinconformers:
                    results[casp_id][key] = []
                    if len(matrix_proteinconformers.keys()) == 4:
                        for m in maps:
                            try:
                                matrix_proteinconformers_aligned = align_and_crop( matrix_proteinconformers[m],matrix_method[m],
                                                                           sequence_proteinconformers, sequence_proteinconformers, )
                                matrix_proteinconformers_aligned, matrix_method_aligned = align_and_crop(
                                        matrix_proteinconformers[m], matrix_method[m], sequence_proteinconformers, sequence_proteinconformers)
                            except:
                                try:
                                    matrix_proteinconformers_aligned, matrix_method_aligned = align_and_crop(
                                        matrix_proteinconformers[m], matrix_method[m], sequence_proteinconformers, sequence_method)
                                except:
                                    traceback.print_exc()
                                    sys.exit(f"{pcpm_method_dp} {m}")
                            # cos_sim = cosine_similarity_across_first_three_channels(matrix_proteinconformers_aligned,matrix_method[m])
                            # print(f"matrix_proteinconformers_aligned: {matrix_proteinconformers_aligned.shape}")
                            # print(f" matrix_method[m]: {matrix_method[m].shape}")
                            cos_sim = rbf_mahalanobis_similarity_first_three_channels(matrix_proteinconformers_aligned, matrix_method_aligned)
                            results[casp_id][key].append(cos_sim.item())
    with open(output_dp,'wb') as o:
        pickle.dump(results,o)

if True:
    method_name = 'esmflow_pdb'
    results = {}
    output_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/results/{method_name}_pcps_rbf_sim.4.pkl"
    for casp_id in tqdm(casp_ids):
        pcpm_method_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/data/{method_name}_new_pcpm_distribution/{method_name}_{casp_id}_pcpmap.pkl"
        pcpm_proteinconformers_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/data/casp1415_new_pcpm_distribution/casp1415_{casp_id}_pcpmap.pkl"
        
        sequence_method = table_df.loc[casp_id]['native_seq']
        sequence_proteinconformers = table_df.loc[casp_id]['casp_seq']
        if os.path.exists(pcpm_method_dp):
            pcpm_method = torch.load(pcpm_method_dp, weights_only=True)
            pcpm_proteinconformers = torch.load(pcpm_proteinconformers_dp,  weights_only=True)
            results[casp_id] = {}
            for key in all_keys:
                # matrix_method = pcpm_method[key]
                matrix_method = pcpm_method['all']
                matrix_proteinconformers = pcpm_proteinconformers[key]
                if matrix_method and matrix_proteinconformers:
                    results[casp_id][key] = []
                    if len(matrix_proteinconformers.keys()) == 4:
                        for m in maps:
                            try:
                                matrix_proteinconformers_aligned = align_and_crop( matrix_proteinconformers[m],matrix_method[m],
                                                                           sequence_proteinconformers, sequence_proteinconformers, )
                                matrix_proteinconformers_aligned, matrix_method_aligned = align_and_crop(
                                        matrix_proteinconformers[m], matrix_method[m], sequence_proteinconformers, sequence_proteinconformers)
                            except:
                                try:
                                    matrix_proteinconformers_aligned, matrix_method_aligned = align_and_crop(
                                        matrix_proteinconformers[m], matrix_method[m], sequence_proteinconformers, sequence_method)
                                except:
                                    traceback.print_exc()
                                    sys.exit(f"{pcpm_method_dp} {m}")
                            # cos_sim = cosine_similarity_across_first_three_channels(matrix_proteinconformers_aligned,matrix_method[m])
                            # print(f"matrix_proteinconformers_aligned: {matrix_proteinconformers_aligned.shape}")
                            # print(f" matrix_method[m]: {matrix_method[m].shape}")
                            cos_sim = rbf_mahalanobis_similarity_first_three_channels(matrix_proteinconformers_aligned, matrix_method_aligned)
                            results[casp_id][key].append(cos_sim.item())
    with open(output_dp,'wb') as o:
        pickle.dump(results,o)

if True:
    method_name = 'bioemu'
    results = {}
    output_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/results/{method_name}_pcps_rbf_sim.4.pkl"
    for casp_id in tqdm(casp_ids):
        pcpm_method_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/data/bioemu_new_pcpm_distribution/{method_name}_{casp_id}_pcpmap.pkl"
        pcpm_proteinconformers_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/data/casp1415_new_pcpm_distribution/casp1415_{casp_id}_pcpmap.pkl"
        
        sequence_method = table_df.loc[casp_id]['native_seq']
        sequence_proteinconformers = table_df.loc[casp_id]['casp_seq']
        if os.path.exists(pcpm_method_dp):
            pcpm_method = torch.load(pcpm_method_dp, weights_only=True)
            pcpm_proteinconformers = torch.load(pcpm_proteinconformers_dp,  weights_only=True)
            results[casp_id] = {}
            for key in all_keys:
                # matrix_method = pcpm_method[key]
                matrix_method = pcpm_method['all']
                matrix_proteinconformers = pcpm_proteinconformers[key]
                if matrix_method and matrix_proteinconformers:
                    results[casp_id][key] = []
                    if len(matrix_proteinconformers.keys()) == 4:
                        for m in maps:
                            try:
                                matrix_proteinconformers_aligned = align_and_crop( matrix_proteinconformers[m],matrix_method[m],
                                                                           sequence_proteinconformers, sequence_proteinconformers, )
                                matrix_proteinconformers_aligned, matrix_method_aligned = align_and_crop(
                                        matrix_proteinconformers[m], matrix_method[m], sequence_proteinconformers, sequence_proteinconformers)
                            except:
                                try:
                                    matrix_proteinconformers_aligned, matrix_method_aligned = align_and_crop(
                                        matrix_proteinconformers[m], matrix_method[m], sequence_proteinconformers, sequence_method)
                                except:
                                    traceback.print_exc()
                                    sys.exit(f"{pcpm_method_dp} {m}")
                            # cos_sim = cosine_similarity_across_first_three_channels(matrix_proteinconformers_aligned,matrix_method[m])
                            # print(f"matrix_proteinconformers_aligned: {matrix_proteinconformers_aligned.shape}")
                            # print(f" matrix_method[m]: {matrix_method[m].shape}")
                            cos_sim = rbf_mahalanobis_similarity_first_three_channels(matrix_proteinconformers_aligned, matrix_method_aligned)
                            results[casp_id][key].append(cos_sim.item())
    with open(output_dp,'wb') as o:
        pickle.dump(results,o)


if True:
    method_name = 'esmdiff'
    results = {}
    output_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/results/{method_name}_pcps_rbf_sim.4.pkl"
    for casp_id in tqdm(casp_ids):
        pcpm_method_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/data/esmdiff_new_pcpm_distribution/{method_name}_{casp_id}_pcpmap.pkl"
        pcpm_proteinconformers_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/data/casp1415_new_pcpm_distribution/casp1415_{casp_id}_pcpmap.pkl"
        if os.path.exists(pcpm_method_dp):
            pcpm_method = torch.load(pcpm_method_dp, weights_only=True)
            pcpm_proteinconformers = torch.load(pcpm_proteinconformers_dp,  weights_only=True)
            results[casp_id] = {}
            for key in all_keys:
                # matrix_method = pcpm_method[key]
                matrix_method = pcpm_method['all']
                matrix_proteinconformers = pcpm_proteinconformers[key]
                if matrix_method and matrix_proteinconformers:
                    results[casp_id][key] = []
                    if len(matrix_proteinconformers.keys()) == 4:
                        for m in maps:
                            try:
                                matrix_proteinconformers_aligned = align_and_crop( matrix_proteinconformers[m],matrix_method[m],
                                                                           sequence_proteinconformers, sequence_proteinconformers, )
                                matrix_proteinconformers_aligned, matrix_method_aligned = align_and_crop(
                                        matrix_proteinconformers[m], matrix_method[m], sequence_proteinconformers, sequence_proteinconformers)
                            except:
                                try:
                                    matrix_proteinconformers_aligned, matrix_method_aligned = align_and_crop(
                                        matrix_proteinconformers[m], matrix_method[m], sequence_proteinconformers, sequence_method)
                                except:
                                    traceback.print_exc()
                                    sys.exit(f"{pcpm_method_dp} {m}")
                            # cos_sim = cosine_similarity_across_first_three_channels(matrix_proteinconformers_aligned,matrix_method[m])
                            # print(f"matrix_proteinconformers_aligned: {matrix_proteinconformers_aligned.shape}")
                            # print(f" matrix_method[m]: {matrix_method[m].shape}")
                            cos_sim = rbf_mahalanobis_similarity_first_three_channels(matrix_proteinconformers_aligned, matrix_method_aligned)
                            results[casp_id][key].append(cos_sim.item())
    with open(output_dp,'wb') as o:
        pickle.dump(results,o)


if True:
    method_name = 'af3'
    results = {}
    output_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/results/{method_name}_pcps_rbf_sim.4.pkl"
    for casp_id in tqdm(casp_ids):
        pcpm_method_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/data/af3_new_pcpm_distribution/{method_name}_{casp_id}_pcpmap.pkl"
        pcpm_proteinconformers_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/data/casp1415_new_pcpm_distribution/casp1415_{casp_id}_pcpmap.pkl"
        if os.path.exists(pcpm_method_dp):
            pcpm_method = torch.load(pcpm_method_dp, weights_only=True)
            pcpm_proteinconformers = torch.load(pcpm_proteinconformers_dp,  weights_only=True)
            results[casp_id] = {}
            for key in all_keys:
                # matrix_method = pcpm_method[key]
                matrix_method = pcpm_method['all']
                matrix_proteinconformers = pcpm_proteinconformers[key]
                if matrix_method and matrix_proteinconformers:
                    results[casp_id][key] = []
                    if len(matrix_proteinconformers.keys()) == 4:
                        for m in maps:
                            try:
                                matrix_proteinconformers_aligned = align_and_crop( matrix_proteinconformers[m],matrix_method[m],
                                                                           sequence_proteinconformers, sequence_proteinconformers, )
                                matrix_proteinconformers_aligned, matrix_method_aligned = align_and_crop(
                                        matrix_proteinconformers[m], matrix_method[m], sequence_proteinconformers, sequence_proteinconformers)
                            except:
                                try:
                                    matrix_proteinconformers_aligned, matrix_method_aligned = align_and_crop(
                                        matrix_proteinconformers[m], matrix_method[m], sequence_proteinconformers, sequence_method)
                                except:
                                    traceback.print_exc()
                                    sys.exit(f"{pcpm_method_dp} {m}")
                            # cos_sim = cosine_similarity_across_first_three_channels(matrix_proteinconformers_aligned,matrix_method[m])
                            # print(f"matrix_proteinconformers_aligned: {matrix_proteinconformers_aligned.shape}")
                            # print(f" matrix_method[m]: {matrix_method[m].shape}")
                            cos_sim = rbf_mahalanobis_similarity_first_three_channels(matrix_proteinconformers_aligned, matrix_method_aligned)
                            results[casp_id][key].append(cos_sim.item())
    with open(output_dp,'wb') as o:
        pickle.dump(results,o)



if True:
    method_name = 'afsample2'
    results = {}
    output_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/results/{method_name}_pcps_rbf_sim.4.pkl"
    for casp_id in tqdm(casp_ids):
        pcpm_method_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/data/afsample2_new_pcpm_distribution/{method_name}_{casp_id}_pcpmap.pkl"
        pcpm_proteinconformers_dp = rf"/mnt/rna01/zyh/prjs/proteinconformers/data/casp1415_new_pcpm_distribution/casp1415_{casp_id}_pcpmap.pkl"
        if os.path.exists(pcpm_method_dp):
            pcpm_method = torch.load(pcpm_method_dp, weights_only=True)
            pcpm_proteinconformers = torch.load(pcpm_proteinconformers_dp,  weights_only=True)
            results[casp_id] = {}
            for key in all_keys:
                # matrix_method = pcpm_method[key]
                matrix_method = pcpm_method['all']
                matrix_proteinconformers = pcpm_proteinconformers[key]
                if matrix_method and matrix_proteinconformers:
                    results[casp_id][key] = []
                    if len(matrix_proteinconformers.keys()) == 4:
                        for m in maps:
                            try:
                                matrix_proteinconformers_aligned = align_and_crop( matrix_proteinconformers[m],matrix_method[m],
                                                                           sequence_proteinconformers, sequence_proteinconformers, )
                                matrix_proteinconformers_aligned, matrix_method_aligned = align_and_crop(
                                        matrix_proteinconformers[m], matrix_method[m], sequence_proteinconformers, sequence_proteinconformers)
                            except:
                                try:
                                    matrix_proteinconformers_aligned, matrix_method_aligned = align_and_crop(
                                        matrix_proteinconformers[m], matrix_method[m], sequence_proteinconformers, sequence_method)
                                except:
                                    traceback.print_exc()
                                    sys.exit(f"{pcpm_method_dp} {m}")
                            # cos_sim = cosine_similarity_across_first_three_channels(matrix_proteinconformers_aligned,matrix_method[m])
                            # print(f"matrix_proteinconformers_aligned: {matrix_proteinconformers_aligned.shape}")
                            # print(f" matrix_method[m]: {matrix_method[m].shape}")
                            cos_sim = rbf_mahalanobis_similarity_first_three_channels(matrix_proteinconformers_aligned, matrix_method_aligned)
                            results[casp_id][key].append(cos_sim.item())
    with open(output_dp,'wb') as o:
        pickle.dump(results,o)



