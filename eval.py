import os, sys
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import cv2

import math
from render_utils import *
from run_nerf_helpers import *

# from load_llff import load_nvidia_data
# import skimage.measure
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity


def im2tensor(image, imtype=np.uint8, cent=1., factor=1./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)
DEBUG = False

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str,default='003615' ,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='logs', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='/data/003615/images',
                        help='input data directory')
    return parser


def calculate_psnr(img1, img2, mask):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mask = mask.astype(np.float64)

    num_valid = np.sum(mask) + 1e-8

    mse = np.sum((img1 - img2)**2 * mask) / num_valid
    
    if mse == 0:
        return 0 #float('inf')

    return 10 * math.log10(1./mse)


def calculate_ssim(img1, img2, mask):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 1]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    _, ssim_map = structural_similarity(img1, img2, multichannel=True, full=True, data_range=1.0,channel_axis=2)
    # _, ssim_map = structural_similarity(img1, img2, multichannel=True, full=True)
    num_valid = np.sum(mask) + 1e-8

    return np.sum(ssim_map * mask) / num_valid



def evaluation():

    parser = config_parser()
    args = parser.parse_args()

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname

    img_dir=os.path.join(basedir, expname)
    img_dir=os.path.join(img_dir, 'testset_300000')

    f = os.path.join(basedir, expname, 'eval.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))


    with torch.no_grad():

        model = models.PerceptualLoss(model='net-lin',net='alex',
                                      use_gpu=True,version=0.1)

        total_psnr = 0.
        total_ssim = 0.
        total_lpips = 0.
        count = 0.
        total_psnr_dy = 0.
        total_ssim_dy = 0.
        total_lpips_dy = 0.
        t = time.time()

        # for each time step
        for img_i in os.listdir(img_dir):
            if 'left' in img_i:
                continue
            rgb_dir=os.path.join(img_dir,img_i)
            gt_i=img_i.replace('png','jpg')
            gt_dir=os.path.join(args.datadir,gt_i)

            rgb = cv2.imread(rgb_dir)[:, :, ::-1]
            # rgb = cv2.resize(rgb, 
            #                     dsize=(rgb.shape[1], rgb.shape[0]), 
            #                     interpolation=cv2.INTER_AREA)
            rgb = np.float32(rgb) / 255

            gt_img = cv2.imread(gt_dir)[:, :, ::-1]
            gt_img = cv2.resize(gt_img, 
                                dsize=(rgb.shape[1], rgb.shape[0]), 
                                interpolation=cv2.INTER_AREA)
            gt_img = np.float32(gt_img) / 255

            psnr = compare_psnr(gt_img, rgb)
            ssim = compare_ssim(gt_img, rgb, channel_axis=2,data_range=1.0)

            gt_img_0 = im2tensor(gt_img).cuda()
            gt_img_0 = im2tensor(gt_img)
            rgb_0 = im2tensor(rgb)
            rgb_0 = im2tensor(rgb).cuda()

            lpips = model.forward(gt_img_0, rgb_0)
            lpips = lpips.item()
            print(psnr, ssim, lpips)
            # print(psnr, ssim)

            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips
            count += 1


            dynamicdir=args.datadir.replace('images','motion_masks')     
            dynamic_mask_path = os.path.join(dynamicdir,
                                            img_i)     
            # print(dynamic_mask_path)
            dynamic_mask = np.float32(cv2.imread(dynamic_mask_path) > 1e-3)#/255.
            dynamic_mask = cv2.resize(dynamic_mask, 
                                    dsize=(rgb.shape[1], rgb.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)

            dynamic_mask_0 = torch.Tensor(dynamic_mask[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

            dynamic_ssim = calculate_ssim(gt_img, 
                                            rgb, 
                                            dynamic_mask)
            dynamic_psnr = calculate_psnr(gt_img, 
                                            rgb, 
                                            dynamic_mask)

            dynamic_lpips = model.forward(gt_img_0, 
                                            rgb_0, 
                                            dynamic_mask_0).item()

            total_psnr_dy += dynamic_psnr
            total_ssim_dy += dynamic_ssim
            total_lpips_dy += dynamic_lpips

        mean_psnr = total_psnr / count
        mean_ssim = total_ssim / count
        mean_lpips = total_lpips / count
        with open(f, 'w') as file:
            file.write('{} = {}\n'.format('mean_psnr', mean_psnr))
            file.write('{} = {}\n'.format('mean_ssim ', mean_ssim))
            file.write('{} = {}\n'.format('mean_lpips ', mean_lpips))


        print('mean_psnr ', mean_psnr) 
        print('mean_ssim ', mean_ssim)
        print('mean_lpips ', mean_lpips)

        mean_psnr_dy = total_psnr_dy / count
        mean_ssim_dy = total_ssim_dy / count
        mean_lpips_dy= total_lpips_dy / count
        
        print('mean_psnr dy', mean_psnr_dy)
        print('mean_ssim dy', mean_ssim_dy)
        print('mean_lpips dy', mean_lpips_dy)


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    evaluation()
