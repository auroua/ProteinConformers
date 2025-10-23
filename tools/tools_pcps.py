# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:12:11 2025

@author: Yihang (Ethan) Zhou


Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

================================ description ==================================

=================================== input =====================================

=================================== output ====================================

================================= parameters ==================================

=================================== example ===================================

=================================== warning ===================================

"""
import os
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
    Align seq1 (length N) and seq2 (length M), then crop matrix1 (4 x N x N)
    to match the aligned residues in seq1 corresponding to seq2.
    N >= M
    """
    assert matrix1.shape[0] == 4 and matrix2.shape[0] == 4
    N = len(seq1)
    M = len(seq2)

    assert matrix1.shape[1] == matrix1.shape[2] == N or matrix1.shape[1] == matrix1.shape[2] == M
    if  matrix1.shape[1] == matrix1.shape[2] == M:
        seq1 = seq2
    assert matrix2.shape[1] == matrix2.shape[2] == M

    # Perform global alignment
    alignment = pairwise2.align.globalxx(seq1, seq2)[0]
    aligned_seq1, aligned_seq2 = alignment.seqA, alignment.seqB

    # Find positions in seq1 that align to real residues (not gaps) in seq2
    aligned_indices = []
    seq1_pos = 0
    seq2_pos = 0
    for a1, a2 in zip(aligned_seq1, aligned_seq2):
        if a1 != '-':
            if a2 != '-':
                aligned_indices.append(seq1_pos)
            seq1_pos += 1
        if a2 != '-':
            seq2_pos += 1

    # Use aligned_indices to slice matrix1
    aligned_indices = np.array(aligned_indices)
    cropped_matrix1 = matrix1[:, aligned_indices][:, :, aligned_indices]

    return cropped_matrix1


def cosine_similarity_across_channels_torch(matrix1: torch.Tensor, matrix2: torch.Tensor) -> torch.Tensor:
    """
    计算两个形状为 (4, N, N) 的张量在每个 (i, j) 点上的通道维度余弦相似度，并对 N×N 平均。

    :param matrix1: torch.Tensor of shape (4, N, N)
    :param matrix2: torch.Tensor of shape (4, N, N)
    :return: scalar tensor, 平均余弦相似度
    """
    assert matrix1.shape == matrix2.shape, "两个矩阵形状必须一致"
    assert matrix1.shape[0] == 4, "矩阵的第一个维度应为 4"

    vec1 = matrix1.view(4, -1).T  # shape: (N*N, 4)
    vec2 = matrix2.view(4, -1).T  # shape: (N*N, 4)

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
    assert matrix1.shape == matrix2.shape, "两个矩阵形状必须一致"
    assert matrix1.dim() == 3 and matrix1.shape[0] == 4, "矩阵应为 (4, N, N)"

    m1 = matrix1[:3]  # shape: (3, N, N)
    m2 = matrix2[:3]  # shape: (3, N, N)

    vec1 = m1.view(3, -1).T  # shape: (N*N, 3)
    vec2 = m2.view(3, -1).T  # shape: (N*N, 3)

    sim = F.cosine_similarity(vec1, vec2, dim=-1)  # shape: (N*N,)

    return sim.mean()

table_dp = r"D:\Projects\protein_design\code\decoys_generation\selected_dataset_734.xlsx"
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
    output_dp = rf"D:\Projects\CASPdynamic\results\{method_name}_pcps_cos_sim.2.pkl"
    for casp_id in tqdm(casp_ids):
        
        tdxm_method_dp = rf"F:\caspdynamics\{method_name}_tdxm_distribution\{method_name}_{casp_id}_tdxmap.pkl"
        tdxm_caspdynamics_dp = rf"F:\caspdynamics\casp_14_15_labels_traintestval_tdxm_pt3\{casp_id}_tdxmap.pt"
        
        sequence_method = table_df.loc[casp_id]['native_seq']
        sequence_caspdynamics = table_df.loc[casp_id]['casp_seq']
        
        tdxm_method = torch.load(tdxm_method_dp, weights_only=True)
        tdxm_caspdynamics = torch.load(tdxm_caspdynamics_dp,  weights_only=True)
        results[casp_id] = {}
        for key in all_keys:
            # matrix_method = tdxm_method[key]
            matrix_method = tdxm_method['all']
            matrix_caspdynamics = tdxm_caspdynamics[key]
            if matrix_method and matrix_caspdynamics:
                results[casp_id][key] = []
                if len(matrix_caspdynamics.keys()) == 4:
                    for m in maps:
                        matrix_caspdynamics_aligned = align_and_crop( matrix_caspdynamics[m],matrix_method[m],
                                                               sequence_caspdynamics, sequence_method, )
                        cos_sim = cosine_similarity_across_first_three_channels(matrix_caspdynamics_aligned,matrix_method[m])
                        results[casp_id][key].append(cos_sim.item())
    with open(output_dp,'wb') as o:
        pickle.dump(results,o)
if True:
    method_name = 'alphaflow_pdb'
    results = {}
    output_dp = rf"D:\Projects\CASPdynamic\results\{method_name}_pcps_cos_sim.2.pkl"
    for casp_id in tqdm(casp_ids):
        
        tdxm_method_dp = rf"F:\caspdynamics\{method_name}_tdxm_distribution\{method_name}_{casp_id}_tdxmap.pkl"
        tdxm_caspdynamics_dp = rf"F:\caspdynamics\casp_14_15_labels_traintestval_tdxm_pt3\{casp_id}_tdxmap.pt"
        
        sequence_method = table_df.loc[casp_id]['native_seq']
        sequence_caspdynamics = table_df.loc[casp_id]['casp_seq']
        
        tdxm_method = torch.load(tdxm_method_dp, weights_only=True)
        tdxm_caspdynamics = torch.load(tdxm_caspdynamics_dp,  weights_only=True)
        results[casp_id] = {}
        for key in all_keys:
            # matrix_method = tdxm_method[key]
            matrix_method = tdxm_method['all']
            matrix_caspdynamics = tdxm_caspdynamics[key]
            if matrix_method and matrix_caspdynamics:
                results[casp_id][key] = []
                if len(matrix_caspdynamics.keys()) == 4:
                    for m in maps:
                        matrix_caspdynamics_aligned = align_and_crop( matrix_caspdynamics[m],matrix_method[m],
                                                               sequence_caspdynamics, sequence_method, )
                        cos_sim = cosine_similarity_across_first_three_channels(matrix_caspdynamics_aligned,matrix_method[m])
                        results[casp_id][key].append(cos_sim.item())
    with open(output_dp,'wb') as o:
        pickle.dump(results,o)

if True:
    method_name = 'esmflow_md'
    results = {}
    output_dp = rf"D:\Projects\CASPdynamic\results\{method_name}_pcps_cos_sim.2.pkl"
    for casp_id in tqdm(casp_ids):
        
        tdxm_method_dp = rf"F:\caspdynamics\{method_name}_tdxm_distribution\{method_name}_{casp_id}_tdxmap.pkl"
        tdxm_caspdynamics_dp = rf"F:\caspdynamics\casp_14_15_labels_traintestval_tdxm_pt3\{casp_id}_tdxmap.pt"
        
        sequence_method = table_df.loc[casp_id]['native_seq']
        sequence_caspdynamics = table_df.loc[casp_id]['casp_seq']
        
        tdxm_method = torch.load(tdxm_method_dp, weights_only=True)
        tdxm_caspdynamics = torch.load(tdxm_caspdynamics_dp,  weights_only=True)
        results[casp_id] = {}
        for key in all_keys:
            # matrix_method = tdxm_method[key]
            matrix_method = tdxm_method['all']
            matrix_caspdynamics = tdxm_caspdynamics[key]
            if matrix_method and matrix_caspdynamics:
                results[casp_id][key] = []
                if len(matrix_caspdynamics.keys()) == 4:
                    for m in maps:
                        matrix_caspdynamics_aligned = align_and_crop( matrix_caspdynamics[m],matrix_method[m],
                                                               sequence_caspdynamics, sequence_method, )
                        cos_sim = cosine_similarity_across_first_three_channels(matrix_caspdynamics_aligned,matrix_method[m])
                        results[casp_id][key].append(cos_sim.item())
    with open(output_dp,'wb') as o:
        pickle.dump(results,o)

if True:
    method_name = 'esmflow_pdb'
    results = {}
    output_dp = rf"D:\Projects\CASPdynamic\results\{method_name}_pcps_cos_sim.2.pkl"
    for casp_id in tqdm(casp_ids):
        
        tdxm_method_dp = rf"F:\caspdynamics\{method_name}_tdxm_distribution\{method_name}_{casp_id}_tdxmap.pkl"
        tdxm_caspdynamics_dp = rf"F:\caspdynamics\casp_14_15_labels_traintestval_tdxm_pt3\{casp_id}_tdxmap.pt"
        
        sequence_method = table_df.loc[casp_id]['native_seq']
        sequence_caspdynamics = table_df.loc[casp_id]['casp_seq']
        
        tdxm_method = torch.load(tdxm_method_dp, weights_only=True)
        tdxm_caspdynamics = torch.load(tdxm_caspdynamics_dp,  weights_only=True)
        results[casp_id] = {}
        for key in all_keys:
            # matrix_method = tdxm_method[key]
            matrix_method = tdxm_method['all']
            matrix_caspdynamics = tdxm_caspdynamics[key]
            if matrix_method and matrix_caspdynamics:
                results[casp_id][key] = []
                if len(matrix_caspdynamics.keys()) == 4:
                    for m in maps:
                        matrix_caspdynamics_aligned = align_and_crop( matrix_caspdynamics[m],matrix_method[m],
                                                               sequence_caspdynamics, sequence_method, )
                        cos_sim = cosine_similarity_across_first_three_channels(matrix_caspdynamics_aligned,matrix_method[m])
                        results[casp_id][key].append(cos_sim.item())
    with open(output_dp,'wb') as o:
        pickle.dump(results,o)

if True:
    method_name = 'bioemu'
    results = {}
    output_dp = rf"D:\Projects\CASPdynamic\results\{method_name}_pcps_cos_sim.2.pkl"
    for casp_id in tqdm(casp_ids):
        
        tdxm_method_dp = rf"F:\caspdynamics\{method_name}_tdxm_distribution\{method_name}_{casp_id}_tdxmap.pkl"
        tdxm_caspdynamics_dp = rf"F:\caspdynamics\casp_14_15_labels_traintestval_tdxm_pt3\{casp_id}_tdxmap.pt"
        
        sequence_method = table_df.loc[casp_id]['native_seq']
        sequence_caspdynamics = table_df.loc[casp_id]['casp_seq']
        
        tdxm_method = torch.load(tdxm_method_dp, weights_only=True)
        tdxm_caspdynamics = torch.load(tdxm_caspdynamics_dp,  weights_only=True)
        results[casp_id] = {}
        for key in all_keys:
            # matrix_method = tdxm_method[key]
            matrix_method = tdxm_method['all']
            matrix_caspdynamics = tdxm_caspdynamics[key]
            if matrix_method and matrix_caspdynamics:
                results[casp_id][key] = []
                if len(matrix_caspdynamics.keys()) == 4:
                    for m in maps:
                        matrix_caspdynamics_aligned = align_and_crop( matrix_caspdynamics[m],matrix_method[m],
                                                               sequence_caspdynamics, sequence_method, )
                        cos_sim = cosine_similarity_across_first_three_channels(matrix_caspdynamics_aligned,matrix_method[m])
                        results[casp_id][key].append(cos_sim.item())
    with open(output_dp,'wb') as o:
        pickle.dump(results,o)

if True:
    method_name = 'esmdiff'
    results = {}
    output_dp = rf"D:\Projects\CASPdynamic\results\{method_name}_pcps_cos_sim.2.pkl"
    for casp_id in tqdm(casp_ids):
        
        tdxm_method_dp = rf"F:\caspdynamics\{method_name}_tdxm_distribution\{method_name}_{casp_id}_tdxmap.pkl"
        tdxm_caspdynamics_dp = rf"F:\caspdynamics\casp_14_15_labels_traintestval_tdxm_pt3\{casp_id}_tdxmap.pt"
        
        if os.path.exists(tdxm_method_dp):
            sequence_method = table_df.loc[casp_id]['native_seq']
            sequence_caspdynamics = table_df.loc[casp_id]['casp_seq']
            
            tdxm_method = torch.load(tdxm_method_dp, weights_only=True)
            tdxm_caspdynamics = torch.load(tdxm_caspdynamics_dp,  weights_only=True)
            results[casp_id] = {}
            for key in all_keys:
                # matrix_method = tdxm_method[key]
                matrix_method = tdxm_method['all']
                matrix_caspdynamics = tdxm_caspdynamics[key]
                if matrix_method and matrix_caspdynamics:
                    results[casp_id][key] = []
                    if len(matrix_caspdynamics.keys()) == 4:
                        for m in maps:
                            matrix_caspdynamics_aligned = align_and_crop( matrix_caspdynamics[m],matrix_method[m],
                                                                   sequence_caspdynamics, sequence_method, )
                            cos_sim = cosine_similarity_across_first_three_channels(matrix_caspdynamics_aligned,matrix_method[m])
                            results[casp_id][key].append(cos_sim.item())
    with open(output_dp,'wb') as o:
        pickle.dump(results,o)













