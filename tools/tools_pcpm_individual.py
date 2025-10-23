# -*- coding: utf-8 -*-
"""
Created on Fri May  2 12:35:00 2025

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
from __future__ import annotations

import argparse
import os
import glob
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

import MDAnalysis as mda
# from MDAnalysis.analysis import distances

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument( "--input_dp", default=r"D:\Projects\CASPdynamic\data\casp_14_15_labels_traintestval.csv", help="big table")
    parser.add_argument( "--output_dp", default=r"D:\Projects\CASPdynamic\data\casp_14_15_labels_traintestval.withXMap.pkl", help="")
    parser.add_argument( "--save_images", type=int, default=0, help="Whether to save heat‑map images (1=yes, 0=no)",)
    parser.add_argument( "--save_tables", type=int, default=0, help="Whether to save CSV tables (1=yes, 0=no)")
    return parser.parse_args()

# =========================================================
# Pairwise metric calculations
# =========================================================

# def compute_distance_map(u: mda.Universe, selection: str = "name CA") -> np.ndarray:
#     """Return NxN CA–CA distance matrix (Å)."""
#     atoms = u.select_atoms(selection)
#     return distances.distance_array(atoms.positions, atoms.positions)


# def compute_contact_map(dist_map: np.ndarray, cutoff: float = 8.0) -> np.ndarray:
#     """Binary contact map: 1 if distance < cutoff Å else 0."""
#     return (dist_map < cutoff).astype(int)


## following are Liyang's code
def sin_cos_angle(p0,p1,p2):
    # [b 3] 
    b0=p0-p1
    b1=p2-p1

    b0=b0 / (torch.norm(b0,dim=-1,keepdim=True)+1e-08)
    b1=b1 / (torch.norm(b1,dim=-1,keepdim=True)+1e-08)
    recos=torch.sum(b0*b1,-1)
    recos=torch.clamp(recos,-0.9999,0.9999)
    resin = torch.sqrt(1-recos**2)
    return resin,recos


def sin_cos_dihedral(p0,p1,p2,p3):

    #p0 = p[:,0:1,:]
    #p1 = p[:,1:2,:]
    #p2 = p[:,2:3,:]
    #p3 = p[:,3:4,:]
    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1=b1/(torch.norm(b1,dim=-1,keepdim=True)+1e-8)  
    
    v = b0 - torch.einsum('bj,bj->b', b0, b1)[:,None]*b1
    w = b2 - torch.einsum('bj,bj->b', b2, b1)[:,None]*b1
    x = torch.einsum('bj,bj->b', v, w)
    #print(x)
    y = torch.einsum('bj,bj->b', torch.cross(b1, v,-1), w)
    #print(y.shape)
    torsion_L = torch.norm(torch.cat([x[:,None],y[:,None]],dim=-1),dim=-1)
    x = x / (torsion_L+1e-8)
    y = y / (torsion_L+1e-8)
    return y,x #torch.atan2(y,x)

def dihedral_2d(p0,p1,p2,p3):
    # p : [L,L,3]
    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2  
    #print(b0.shape)
    b1=b1/(torch.norm(b1,dim=-1,keepdim=True)+1e-8) 
    v = b0 - torch.einsum('abj,abj->ab', b0, b1)[...,None]*b1
    w = b2 - torch.einsum('abj,abj->ab', b2, b1)[...,None]*b1
    x = torch.einsum('abj,abj->ab', v, w)
    y = torch.einsum('abj,abj->ab', torch.cross(b1, v,-1), w)
    return torch.atan2(y,x)
def dihedral_1d(p0,p1,p2,p3):
    # p : [L,L,3]
    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2  
    # print(b0.shape)
    b1=b1/(torch.norm(b1,dim=-1,keepdim=True)+1e-8) 
    v = b0 - torch.einsum('bj,bj->b', b0, b1)[...,None]*b1
    w = b2 - torch.einsum('bj,bj->b', b2, b1)[...,None]*b1
    x = torch.einsum('bj,bj->b', v, w)
    y = torch.einsum('bj,bj->b', torch.cross(b1, v,-1), w)
    return torch.atan2(y,x)

def angle_2d(p0,p1,p2):
    # [a b 3] 
    b0=p0-p1
    b1=p2-p1
    b0=b0 / (torch.norm(b0,dim=-1,keepdim=True)+1e-08)
    b1=b1 / (torch.norm(b1,dim=-1,keepdim=True)+1e-08)
    recos=torch.sum(b0*b1,-1)
    recos=torch.clamp(recos,-0.9999,0.9999)
    return torch.arccos(recos)

def angle_1d(p0,p1,p2):
    return angle_2d(p0,p1,p2)

def distance_2d(p0,p1):
    return  (p0-p1).norm(dim=-1)

def get_omg_map(x):
    # x: L 4 3    N CA C CB
    L=x.shape[0]
    cai=x[:,None,1].repeat(1,L,1)
    cbi=x[:,None,-1].repeat(1,L,1)
    cbj=x[None,:,-1].repeat(L,1,1)
    caj=x[None,:,1].repeat(L,1,1)
    torsion = dihedral_2d(cai,cbi,cbj,caj)
    return torsion

def get_phi_map(x):
    L=x.shape[0]
    cai=x[:,None,1].repeat(1,L,1) 
    cbi=x[:,None,-1].repeat(1,L,1) 
    cbj=x[None,:,-1].repeat(L,1,1)
    return angle_2d(cai,cbi,cbj)

def get_theta_map(x):
    L=x.shape[0]
    ni =x[:,None,0].repeat(1,L,1)
    cai=x[:,None,1].repeat(1,L,1) 
    cbi=x[:,None,-1].repeat(1,L,1) 
    cbj=x[None,:,-1].repeat(L,1,1)
    return dihedral_2d(ni,cai,cbi,cbj)

def get_cadis_map(x):
    cai=x[:,None,1]
    caj=x[None,:,1]
    return distance_2d(cai,caj)

def get_cbdis_map(x):
    cai=x[:,None,-1]
    caj=x[None,:,-1]
    return distance_2d(cai,caj)


def get_all(x):
    L=x.shape[0]
    ni =x[:,None,0].repeat(1,L,1)
    cai=x[:,None,1].repeat(1,L,1)
    ci= x[:,None,2].repeat(1,L,1)
    cbi=x[:,None,-1].repeat(1,L,1)

    nj =x[None,:,0].repeat(L,1,1) 
    caj=x[None,:,1].repeat(L,1,1)  
    cj =x[None,:,2].repeat(L,1,1)    
    cbj=x[None,:,-1].repeat(L,1,1)

    cbmap=distance_2d(cbi,cbj)
    camap=distance_2d(cai,caj)
    ncmap=distance_2d(ni,cj)

    omgmap=dihedral_2d(cai,cbi,cbj,caj)
    psimap=angle_2d(cai,cbi,cbj)
    thetamap=dihedral_2d(ni,cai,cbi,cbj)

    
    canccamap=dihedral_2d(cai,ni,cj,caj)
    cancmap=angle_2d(cai,ni,cj)
    cacnmap=angle_2d(cai,ci,nj)

    return cbmap,camap,  omgmap,psimap,thetamap,  ncmap,canccamap,cancmap,cacnmap

def virtual_CB(n, ca, c):
    # MDTraj 自带函数：md.geometry.geometry.virtual_center(...) 也行
    b = ca - n
    c_vect = c - ca
    a = np.cross(b, c_vect)
    vcb = ca + (-0.58273431 * b) + (0.56802827 * c_vect) + (0.54067466 * a / np.linalg.norm(a))
    return vcb

def pdb_to_true_x(pdb_path: str, atom_order=['N', 'CA', 'C', 'CB']) -> torch.Tensor:
    u = mda.Universe(pdb_path)
    residues = u.select_atoms("protein and name N CA C CB").residues
    coords = []
    for res in residues:
        atoms = []
        for atom_name in atom_order:
            atom = res.atoms.select_atoms(f"name {atom_name}")
            if atom.n_atoms > 0:
                atoms.append(atom.positions[0])
            else:
                # atoms.append(np.full(3, np.nan))  # 缺失时补 NaN
                atoms.append(virtual_CB(*res.atoms.select_atoms("name N CA C").positions))
        coords.append(atoms)
    coords = np.array(coords)  # shape = [L, 4, 3]
    return torch.tensor(coords, dtype=torch.float32)

def torch_to_np16_upper(t: torch.Tensor) -> np.ndarray:
    """
    把 2D 对称矩阵的 torch.Tensor
    → numpy.float16
    → 仅保留上三角(含对角)并拉平成 1‑D 向量
    """
    arr = t.to(torch.float16).cpu().numpy()
    idx = np.triu_indices(arr.shape[0])
    return arr[idx]          # shape = [L*(L+1)//2]

def tri_to_square(vec, L, keep_diag=True, dtype=np.float16):
    k = 0 if keep_diag else 1
    mat = np.zeros((L, L), dtype=dtype)
    idx = np.triu_indices(L, k=k)
    mat[idx] = vec
    mat = mat + mat.T
    if not keep_diag:
        mat[np.diag_indices(L)] = 0
    return mat

#### Main
if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(args.input_dp)
    for casp_id, df_sub in df.groupby("casp_id"):
        base, _ = os.path.splitext(args.output_dp)          # 去掉末尾 .pkl（若有）
        out_file = f"{base}_{casp_id}.pkl"                  # 加上 casp_id
        if os.path.exists(out_file):
            continue
        else:
            maps_dict = {}
            print(f"{casp_id} is processing.\n")
            for idx, row in tqdm(df_sub.iterrows(), total=len(df_sub), desc="Processing decoys"):
                maps_dict_tmp = {}
                decoy_id = row["decoy"]  # ensure basename only
                gromacs_dir = decoy_id.split("_gromacs")[0]
            
                pdb_path = f"/mnt/rna01/zyh/data/selected_database2/{casp_id}/decoys_MD/{gromacs_dir}_gromacs/{decoy_id}"
            
                true_x = pdb_to_true_x(pdb_path)
                # pair‑wise maps (1×N×N each)
                cbdis_map,cadis_map,  omg_map,phi_map,theta_map,  nc_map,cancca_map,canc_map,cacn_map = get_all(true_x)
        
                # accumulate
                maps_dict_tmp['length'] = true_x.shape[0]
                # maps_dict_tmp["cbdis"] = torch_to_np16_upper(cbdis_map)
                maps_dict_tmp["cadis"] = torch_to_np16_upper(cadis_map)
                maps_dict_tmp["omg"] = torch_to_np16_upper(omg_map)
                maps_dict_tmp["phi"] = torch_to_np16_upper(phi_map)
                maps_dict_tmp["theta"] = torch_to_np16_upper(theta_map)
                # maps_dict_tmp["nc"] = torch_to_np16_upper(nc_map)
                # maps_dict_tmp["cancca"] = torch_to_np16_upper(cancca_map)
                # maps_dict_tmp["canc"] = torch_to_np16_upper(canc_map)
                # maps_dict_tmp["cacn"] = torch_to_np16_upper(cacn_map)
                maps_dict[f"{casp_id}_{decoy_id}"] = maps_dict_tmp
            with open(out_file, "wb") as o:
                pickle.dump(maps_dict, o)



