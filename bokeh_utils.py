import os
import cv2
import time
import imageio
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def MPIBokehRenderer_final(raw,alpha,weight=None,num_pt=None):
    rgb = torch.sigmoid(raw[...,:3])
    mpi_image = rgb

    mpi_image = torch.cat((mpi_image, torch.ones_like(alpha[...,None])), dim=2)
    alpha_new=alpha

    pt_num=num_pt
    mpi_pts = mpi_image.reshape(-1, pt_num,128, 4)
    alpha_pts = alpha_new.reshape(-1, pt_num,128, 1)
    mpi_bokeh= torch.sum(mpi_pts *alpha_pts* weight[..., None,None], dim=1)
    mpi_alpha=torch.sum(alpha_pts* weight[..., None,None], dim=1).squeeze(-1)
    weights_new = torch.cumprod(torch.cat([torch.ones((mpi_alpha.shape[0], 1)), 1.-mpi_alpha + 1e-10], -1), -1)[:, :-1]
    bokeh_final = torch.sum(weights_new[...,None] * mpi_bokeh, -2) ## N_rays,4
    # norm
    bokeh_final = bokeh_final[:, :-1] / bokeh_final[:, -1:]
    
    return bokeh_final
def MPIBokehRenderer_blending_final(raw_dy,raw_rigid,alpha_dy,alpha_rig,weight,num_pt):
    rgb_dy = torch.sigmoid(raw_dy[...,:3])
    mpi_image_dy = rgb_dy 
    mpi_image_dy = torch.cat((mpi_image_dy, torch.ones_like(alpha_dy[...,None])), dim=2)

    rgb_rigid = torch.sigmoid(raw_rigid[...,:3])
    mpi_image_rig = rgb_rigid 
    mpi_image_rig = torch.cat((mpi_image_rig, torch.ones_like(alpha_rig[...,None])), dim=2)

    
    pt_num=num_pt
    mpi_dy_pts = mpi_image_dy.reshape(-1, pt_num,128, 4)
    alpha_dy_pts = alpha_dy.reshape(-1, pt_num,128, 1)
    mpi_bokeh_dy= torch.sum(mpi_dy_pts *alpha_dy_pts* weight[..., None,None], dim=1)
    mpi_alpha_dy=torch.sum(alpha_dy_pts* weight[..., None,None], dim=1).squeeze(-1)

    mpi_rig_pts = mpi_image_rig.reshape(-1, pt_num,128, 4)
    alpha_rig_pts = alpha_rig.reshape(-1, pt_num,128, 1)
    mpi_bokeh_rig= torch.sum(mpi_rig_pts *alpha_rig_pts* weight[..., None,None], dim=1)
    mpi_alpha_rig=torch.sum(alpha_rig_pts* weight[..., None,None], dim=1).squeeze(-1)

    
    
    Ts_new = torch.cumprod(torch.cat([torch.ones((mpi_alpha_dy.shape[0], 1)), 
                                (1. - mpi_alpha_dy) * (1. - mpi_alpha_rig)  + 1e-10], -1), -1)[:, :-1]
    weights_dy = Ts_new 
    weights_rig = Ts_new
    bokeh_final = torch.sum(weights_dy[..., None] * mpi_bokeh_dy + \
                        weights_rig[..., None] * mpi_bokeh_rig, -2) 
    # norm
    bokeh_final = bokeh_final[:, :-1] /(bokeh_final[:, -1:]+1e-8) 

    
    return bokeh_final
