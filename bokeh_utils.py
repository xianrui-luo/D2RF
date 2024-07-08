import os
import cv2
import time
import imageio
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn
# from bokeh_renderer.scatter_ex import ModuleRenderScatterEX as ModuleRenderScatter
# from scatter import ModuleRenderScatter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SoftDiskBlur(nn.Module):
    def __init__(self, kernel_size):
        super(SoftDiskBlur, self).__init__()
        self.r = kernel_size // 2
        x_grid, y_grid = torch.meshgrid(torch.arange(-int(self.r), int(self.r)+1), torch.arange(-int(self.r), int(self.r)+1))
        # kernel = (x_grid**2 + y_grid**2) <= r**2
        kernel = 0.5 + 0.5 * torch.tanh(0.25 * (self.r**2 - x_grid**2 - y_grid**2) + 0.5)
        kernel = kernel.to(torch.float32) / kernel.sum()
        # kernel = kernel.expand(3, 1, kernel_size, kernel_size)
        kernel = kernel.expand(3, 1, kernel_size, kernel_size)
        # self.pad = nn.ReflectionPad2d(r)  # mirror fill
        self.weight = kernel.to(device)
 
    def forward(self, x):
        # out = self.pad(x)
        out=F.pad(x,pad=(self.r,self.r,self.r,self.r),mode='replicate')
        ch = x.shape[1]
        out = F.conv2d(out, self.weight[:ch], padding=0, groups=ch)
        return out
class SoftDiskBlur_mpi(nn.Module):
    def __init__(self, kernel_size):
        super(SoftDiskBlur_mpi, self).__init__()
        r = kernel_size // 2
        x_grid, y_grid = torch.meshgrid(torch.arange(-int(r), int(r)+1), torch.arange(-int(r), int(r)+1))
        kernel = 0.5 + 0.5 * torch.tanh(0.25 * (r**2 - x_grid**2 - y_grid**2) + 0.5)
        kernel = kernel / kernel.sum()
        kernel = kernel.expand(1, 1, kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.pad = nn.ReplicationPad2d(r)
        self.weight = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        out = self.pad(x)
        ch = x.shape[1]
        out = F.conv2d(out, self.weight.expand(ch, 1, self.kernel_size, self.kernel_size), padding=0, groups=ch)
        return out

    
# def gaussian_blur(x, r, sigma=None):
#     r = int(round(r))
#     if sigma is None:
#         sigma = 0.3 * (r - 1) + 0.8
#     x_grid, y_grid = torch.meshgrid(torch.arange(-int(r), int(r) + 1), torch.arange(-int(r), int(r) + 1))
#     kernel = torch.exp(-(x_grid ** 2 + y_grid ** 2) / 2 / sigma ** 2)
#     kernel = kernel.float() / kernel.sum()
#     kernel = kernel.expand(1, 1, 2*r+1, 2*r+1).to(x.device)
#     x = F.pad(x, pad=(r, r, r, r), mode='replicate')
#     x = F.conv2d(x, weight=kernel, padding=0)
#     return x



# def render_bokeh(rgbs, 
#                 disps,
#                 K_bokeh=20, 
#                 gamma=4, 
#                 disp_focus=90/255, 
#                 defocus_scale=1):
    
#     # classical_renderer = ModuleRenderScatter().to(device)

#     # disps =  (disps - disps.min()) / (disps.ma x()- disps.min())
#     # disps = disps / disps.max()
    
#     signed_disp = disps - disp_focus
#     defocus = torch.abs(signed_disp) / defocus_scale

#     defocus = defocus.unsqueeze(0).unsqueeze(0).contiguous()
#     rgbs = rgbs.permute(2, 0, 1).unsqueeze(0).contiguous()

#     # bokeh_classical = classical_renderer(rgbs**gamma, defocus*defocus_scale)
#     bokeh_classical = to_blur_single(rgbs**gamma, defocus,K_bokeh,disp_focus)
#     bokeh_classical = bokeh_classical ** (1/gamma)
#     bokeh_classical = bokeh_classical[0].permute(1, 2, 0)
#     return bokeh_classical


def MPIBokehRenderer(raw,alpha,z_vals, K, disp_focus,blur_kernels):
    gamma=2.2
    # raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    # N_pixel = 48
    N_pixel = int(np.sqrt(z_vals.shape[0]))
    # disp_map=disp_map/5.0 ##max_disp
    # disp_map=torch.reshape(disp_map,[N_pixel,N_pixel,-1])
    z_vals=torch.reshape(z_vals,[N_pixel,N_pixel,-1])
    signed_disp=z_vals-disp_focus
    defocus = abs(signed_disp)
    
    
    rgb = torch.sigmoid(raw[...,:3])
    # mpi_image = rgb ** gamma
    mpi_image = rgb
    # 
    # mpi_image=rgb
    mpi_image = torch.cat((mpi_image, torch.ones_like(alpha.unsqueeze(2))), dim=2)

    ## no norm
    # alpha_norm=torch.ones_like(raw[...,3]).unsqueeze(-1)
    # mpi_image = torch.cat((mpi_image, alpha_norm), dim=2)
    ##mpi_image shape 48 48 128 3+1

    
    # bokeh = torch.zeros_like(mpi_image[:, 0])
    bins=int(round(float(torch.max(defocus[0,0,:] * K))))
    ##增大focus range
    # defocus = defocus - 0.5 / 32
    # defocus=torch.where(defocus>0,defocus,torch.zeros_like(defocus))
    # defocus[0,0,:]=torch.where(defocus[0,0,:]>0,defocus[0,0,:],torch.zeros_like(defocus[0,0,:]))
    ##shape 48 48 128
    # kernels = nn.ModuleList()
    kernel=[]
    # mpi_image=torch.reshape(mpi_image,[N_pixel,N_pixel,-1,3])
    mpi_image=torch.reshape(mpi_image,[N_pixel,N_pixel,-1,4])
    alpha_new=torch.reshape(alpha.clone(),[N_pixel,N_pixel,-1])
    mpi_image=mpi_image.permute(2,3,0,1) ###N,C,H,W
    alpha_new=alpha_new.permute(2,0,1) ###N,C,H,W
    mpi_image_list=torch.zeros_like(mpi_image)
    alpha_list=torch.zeros_like(alpha_new)
    
    
    for i in range(bins+1):
        # d = i * 1 / (bins - 1)
        # mask = 1/2 + 1/2 * torch.tanh(100 * (0.5- (defocus[0,0,:] * K - i)))
        # mask = 1/2 + 1/2 * torch.tanh(100 * (1 / (bins - 1)- torch.abs(defocus[0,0,:] * K - d)))
        # mask=torch.where(torch.round(defocus * K).to(torch.uint8)==int(i),torch.ones_like(defocus),torch.zeros_like(defocus))
        # mask=torch.where(torch.round(defocus[0,0,:] * K).to(torch.uint8)==int(i),torch.ones_like(defocus[0,0,:]),torch.zeros_like(defocus[0,0,:]))
        # mask=(mask>0.5)
        # if (mask==True).sum()==0:
        #     continue
        # mindefocus= defocus[0,0,:].argmin()
        # kernel=blur_kernels[int(torch.round(float(defocus[0,0,i] * K)))]
        # if i!=0:
        #     kernel=SoftDiskBlur_mpi(int(2*i+1)).to(device)

        # else:
        #     kernel=blur_kernels[0]
            # kernel=nn.Identity().to(device)
        defocus_final=defocus*K
        mask = 1/2 + 1/2 * torch.tanh(500 * (0.5- torch.abs(defocus_final[0,0,:]- i)))
        kernel=blur_kernels[i]
        mask=mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(mask.shape[0],1,N_pixel,N_pixel)

        # mpi_image=kernel(mpi_image*alpha_new[None])
        # mpi_image_list.append(kernel(mpi_image))
        # alpha_list.append(kernel(alpha_new[None]))
        mpi_image_list=mpi_image_list+kernel(mpi_image*mask*alpha_new[:,None])
        alpha_list[:,None]=alpha_list[:,None]+kernel(alpha_new[:,None]*mask)
        

    # 128/4=32个kernel 
        

    mpi_image=mpi_image_list.permute(2,3,0,1) 
    # mpi_image=torch.reshape(mpi_image,[N_pixel*N_pixel,-1,3])
    mpi_image=torch.reshape(mpi_image,[N_pixel*N_pixel,-1,4])
    # alpha_new=alpha_new.permute(2,0,1) 
    alpha_new=alpha_list.permute(1,2,0) ###N,C,H,W
    alpha_new=torch.reshape(alpha_new,[N_pixel*N_pixel,-1])
    
    
    # weights_new = alpha_new * torch.cumprod(torch.cat([torch.ones((alpha_new.shape[0], 1)), 1.-alpha_new+ 1e-10], -1), -1)[:, :-1]
    weights_new = torch.cumprod(torch.cat([torch.ones((alpha_new.shape[0], 1)), 1.-alpha_new + 1e-10], -1), -1)[:, :-1]
    bokeh_final = torch.sum(weights_new[...,None] * mpi_image, -2) ## N_rays,4
    # norm
    bokeh_final = bokeh_final[:, :-1] / bokeh_final[:, -1:]

    ## need degamma but outside the GT
    # bokeh_final = bokeh_final.clamp(1e-10, 1e10) ** (1 / gamma)
    # bokeh_final = bokeh_final ** (1 / gamma)

    
    return bokeh_final
def MPIBokehRenderer_final(raw,alpha,z_vals, K=5, disp_focus=0.5,weight=None,num_pt=None):
    # gamma=2.2
    # N_pixel = int(np.sqrt(z_vals.shape[0]))
    # z_vals=torch.reshape(z_vals,[N_pixel,N_pixel,-1])
    # signed_disp=z_vals-disp_focus
    # defocus = abs(signed_disp)
    rgb = torch.sigmoid(raw[...,:3])
    # mpi_image = rgb ** gamma
    mpi_image = rgb

    mpi_image = torch.cat((mpi_image, torch.ones_like(alpha[...,None])), dim=2)
    alpha_new=alpha

    pt_num=num_pt
    mpi_pts = mpi_image.reshape(-1, pt_num,128, 4)
    alpha_pts = alpha_new.reshape(-1, pt_num,128, 1)
    mpi_bokeh= torch.sum(mpi_pts *alpha_pts* weight[..., None,None], dim=1)
    mpi_alpha=torch.sum(alpha_pts* weight[..., None,None], dim=1).squeeze(-1)
    
    # mpi_image=mpi_image_list.permute(2,3,0,1) 
    # mpi_image=torch.reshape(mpi_image,[N_pixel*N_pixel,-1,3])
    # mpi_image=torch.reshape(mpi_image,[N_pixel*N_pixel,-1,4])
    # alpha_new=alpha_new.permute(2,0,1) 
    # alpha_new=alpha_list.permute(1,2,0) ###N,C,H,W
    # alpha_new=torch.reshape(alpha_new,[N_pixel*N_pixel,-1])
    # weights_new = torch.cumprod(torch.cat([torch.ones((alpha_new.shape[0], 1)), 1.-alpha_new + 1e-10], -1), -1)[:, :-1]
    weights_new = torch.cumprod(torch.cat([torch.ones((mpi_alpha.shape[0], 1)), 1.-mpi_alpha + 1e-10], -1), -1)[:, :-1]
    bokeh_final = torch.sum(weights_new[...,None] * mpi_bokeh, -2) ## N_rays,4
    # norm
    bokeh_final = bokeh_final[:, :-1] / bokeh_final[:, -1:]
    
    return bokeh_final
def MPIBokehRenderer_blending_final(raw_dy,raw_rigid,alpha_dy,alpha_rig,raw_blend_w,z_vals, K, disp_focus,weight,num_pt):
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
    # weights_new = alpha_new * torch.cumprod(torch.cat([torch.ones((alpha_new.shape[0], 1)), 1.-alpha_new+ 1e-10], -1), -1)[:, :-1]
    # weights_dy = Ts_new * alpha_dy_new
    weights_dy = Ts_new 
    # weights_rig = Ts_new * alpha_rig_new
    weights_rig = Ts_new
    bokeh_final = torch.sum(weights_dy[..., None] * mpi_bokeh_dy + \
                        weights_rig[..., None] * mpi_bokeh_rig, -2) 
    # norm
    bokeh_final = bokeh_final[:, :-1] /(bokeh_final[:, -1:]+1e-8) 

    
    return bokeh_final
def MPIBokehRenderer_blending(raw_dy,raw_rigid,alpha_dy,alpha_rig,raw_blend_w,z_vals, K, disp_focus,blur_kernels):
    gamma=2.2
    # raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    # N_pixel = 48
    N_pixel = int(np.sqrt(z_vals.shape[0]))
    z_vals=torch.reshape(z_vals,[N_pixel,N_pixel,-1])
    signed_disp=z_vals-disp_focus
    defocus = abs(signed_disp)
    
    
    rgb_dy = torch.sigmoid(raw_dy[...,:3])
    mpi_image_dy = rgb_dy 
    mpi_image_dy = torch.cat((mpi_image_dy, torch.ones_like(alpha_dy.unsqueeze(2))), dim=2)

    rgb_rigid = torch.sigmoid(raw_rigid[...,:3])
    mpi_image_rig = rgb_rigid 
    mpi_image_rig = torch.cat((mpi_image_rig, torch.ones_like(alpha_rig.unsqueeze(2))), dim=2)
    

    bins=int(round(float(torch.max(defocus[0,0,:] * K))))
    ##增大focus range
    # defocus = defocus - 0.5 / 32
    # defocus=torch.where(defocus>0,defocus,torch.zeros_like(defocus))
    kernel=[]
    # mpi_image_dy=torch.reshape(mpi_image_dy,[N_pixel,N_pixel,-1,3])
    mpi_image_dy=torch.reshape(mpi_image_dy,[N_pixel,N_pixel,-1,4])
    alpha_dy_new=torch.reshape(alpha_dy.clone(),[N_pixel,N_pixel,-1])
    # mpi_image_rig=torch.reshape(mpi_image_rig,[N_pixel,N_pixel,-1,3])
    mpi_image_rig=torch.reshape(mpi_image_rig,[N_pixel,N_pixel,-1,4])
    alpha_rig_new=torch.reshape(alpha_rig.clone(),[N_pixel,N_pixel,-1])
    mpi_image_dy=mpi_image_dy.permute(2,3,0,1) ###N,C,H,W
    alpha_dy_new=alpha_dy_new.permute(2,0,1) ###N,C,H,W
    mpi_image_rig=mpi_image_rig.permute(2,3,0,1) ###N,C,H,W
    alpha_rig_new=alpha_rig_new.permute(2,0,1) ###N,C,H,W

    mpi_dy_list=torch.zeros_like(mpi_image_dy)
    mpi_rig_list=torch.zeros_like(mpi_image_rig)
    alpha_dy_list=torch.zeros_like(alpha_dy_new)
    alpha_rig_list=torch.zeros_like(alpha_rig_new)
    
    
    for i in range(bins+1):
        # d = i * 1 / (bins - 1)
        # mask = 1/2 + 1/2 * torch.tanh(100 * (1 / (bins - 1)- torch.abs(defocus[0,0,:] - d)))
        # mask=torch.where(torch.round(defocus * K).to(torch.uint8)==int(i),torch.ones_like(defocus),torch.zeros_like(defocus))
        # mask=torch.where(torch.round(defocus[0,0,:] * K).to(torch.uint8)==int(i),torch.ones_like(defocus[0,0,:]),torch.zeros_like(defocus[0,0,:]))
        # mask=(mask>0.5)
        # if (mask==True).sum()==0:
        #     continue
        # # mindefocus= defocus[0,0,:].argmin()
        # kernel=blur_kernels[int(torch.round(defocus[0,0,i] * K))]
        # if i!=0:
        #     kernel=SoftDiskBlur_mpi(int(2*i+1)).to(device)

        # else:
        #     kernel=nn.Identity().to(device)
        defocus_final=defocus*K
        mask = 1/2 + 1/2 * torch.tanh(500 * (0.5- torch.abs(defocus_final[0,0,:]- i)))
        kernel=blur_kernels[i]
        mask=mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(mask.shape[0],1,N_pixel,N_pixel)

        mpi_dy_list=mpi_dy_list+kernel(mpi_image_dy*mask*alpha_dy_new[:,None])
        mpi_rig_list=mpi_rig_list+kernel(mpi_image_rig*mask*alpha_rig_new[:,None])
        alpha_dy_list[:,None]=alpha_dy_list[:,None]+kernel(alpha_dy_new[:,None]*mask)
        alpha_rig_list[:,None]=alpha_rig_list[:,None]+kernel(alpha_rig_new[:,None]*mask)
        # alpha_dy_new[mask,None]=kernel(alpha_dy_new[mask,None])
        # mpi_image_rig[mask]=kernel(mpi_image_rig[mask]*alpha_rig_new[mask,None])
        # alpha_rig_new[mask,None]=kernel(alpha_rig_new[mask,None])

    # 128/4=32个kernel 
    mpi_image_dy=mpi_dy_list.permute(2,3,0,1) 
    # mpi_image_dy=torch.reshape(mpi_image_dy,[N_pixel*N_pixel,-1,3])
    mpi_image_dy=torch.reshape(mpi_image_dy,[N_pixel*N_pixel,-1,4])
    alpha_dy_new=alpha_dy_list.permute(1,2,0) ###N,C,H,W
    alpha_dy_new=torch.reshape(alpha_dy_new,[N_pixel*N_pixel,-1])
    mpi_image_rig=mpi_rig_list.permute(2,3,0,1) 
    # mpi_image_rig=torch.reshape(mpi_image_rig,[N_pixel*N_pixel,-1,3])
    mpi_image_rig=torch.reshape(mpi_image_rig,[N_pixel*N_pixel,-1,4])
    alpha_rig_new=alpha_rig_list.permute(1,2,0) ###N,C,H,W
    alpha_rig_new=torch.reshape(alpha_rig_new,[N_pixel*N_pixel,-1])
    
    Ts_new = torch.cumprod(torch.cat([torch.ones((alpha_dy.shape[0], 1)), 
                                (1. - alpha_dy_new) * (1. - alpha_rig_new)  + 1e-10], -1), -1)[:, :-1]
    # weights_new = alpha_new * torch.cumprod(torch.cat([torch.ones((alpha_new.shape[0], 1)), 1.-alpha_new+ 1e-10], -1), -1)[:, :-1]
    # weights_dy = Ts_new * alpha_dy_new
    weights_dy = Ts_new 
    # weights_rig = Ts_new * alpha_rig_new
    weights_rig = Ts_new
    bokeh_final = torch.sum(weights_dy[..., None] * mpi_image_dy + \
                        weights_rig[..., None] * mpi_image_rig, -2) 
    # norm
    bokeh_final = bokeh_final[:, :-1] / (bokeh_final[:, -1:]+1e-8) 

    ## need degamma but outside the GT
    # bokeh_final = bokeh_final.clamp(1e-10, 1e10) ** (1 / gamma)
    # bokeh_final = bokeh_final ** (1 / gamma)

    
    return bokeh_final
