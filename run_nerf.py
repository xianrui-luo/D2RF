import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import cv2
from kornia import create_meshgrid

from render_utils_mpi_final import *
from run_nerf_helpers_mpi import *
from load_llff_mpi import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)
DEBUG = False

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',

                        help='input data directory')
    parser.add_argument("--render_lockcam_slowmo", action='store_true', 
                        help='render fixed view + slowmo')
    parser.add_argument("--render_slowmo_bt", action='store_true', 
                        help='render space-time interpolation')

    parser.add_argument("--final_height", type=int, default=288, 
                        help='training image height, default is 512x288')
    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=300, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*128, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*128, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_bt", action='store_true', 
                        help='render bullet time')

    parser.add_argument("--render_test", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    parser.add_argument("--target_idx", type=int, default=10, 
                        help='target_idx')
    parser.add_argument("--num_extra_sample", type=int, default=512, 
                        help='num_extra_sample')
    parser.add_argument("--decay_depth_w", action='store_true', 
                        help='decay depth weights')
    parser.add_argument("--use_motion_mask", action='store_true', 
                        help='use motion segmentation mask for hard-mining data-driven initialization')
    parser.add_argument("--decay_optical_flow_w", action='store_true', 
                        help='decay optical flow weights')

    parser.add_argument("--w_depth",   type=float, default=0.04, 
                        help='weights of depth loss')
    parser.add_argument("--w_optical_flow", type=float, default=0.02, 
                        help='weights of optical flow loss')
    parser.add_argument("--w_sm", type=float, default=0.1, 
                        help='weights of scene flow smoothness')
    parser.add_argument("--w_sf_reg", type=float, default=0.1, 
                        help='weights of scene flow regularization')
    parser.add_argument("--w_cycle", type=float, default=0.1, 
                        help='weights of cycle consistency')
    parser.add_argument("--w_prob_reg", type=float, default=0.1, 
                        help='weights of disocculusion weights')

    parser.add_argument("--w_entropy", type=float, default=1e-3, 
                        help='w_entropy regularization weight')

    parser.add_argument("--decay_iteration", type=int, default=50, 
                        help='data driven priors decay iteration * 1000')

    parser.add_argument("--chain_sf", action='store_true', 
                        help='5 frame consistency if true, \
                             otherwise 3 frame consistency')

    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=50)

    parser.add_argument("--N_iters", type=int, default=200000)
    parser.add_argument("--bokeh_iters", type=int, default=200000+1)
    parser.add_argument("--num_pt", type=int, default=5)
    parser.add_argument("--kernel_window", type=int, default=10)

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=1000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=1000, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_testset",     type=int, default=50000, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=5000, 
                        help='frequency of weight ckpt saving')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    if args.dataset_type == 'llff':
        target_idx = args.target_idx
        images, depths, masks, poses, bds, \
        render_poses, ref_c2w, motion_coords,bg_masks = load_llff_data(args.datadir, 
                                                            args.start_frame, args.end_frame,
                                                            args.factor,
                                                            target_idx=target_idx,
                                                            recenter=True, bd_factor=.9,
                                                            spherify=args.spherify, 
                                                            final_height=args.final_height)

        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        ##modified by lxr
        # i_test = []
        i_val = [] #i_test
        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            # i_test = np.arange(images.shape[0])[::args.llffhold]
            i_test = np.arange(images.shape[0])[1::args.llffhold]


        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])
        


        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.percentile(bds[:, 0], 5) * 0.8 #np.ndarray.min(bds) #* .9
            far = np.percentile(bds[:, 1], 95) * 1.1 #np.ndarray.max(bds) #* 1.
        else:
            near = 0.
            far = 1.

        print('NEAR FAR', near, far)
    else:
        print('ONLY SUPPORT LLFF!!!!!!!!')
        sys.exit()



    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    K=None
    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    # Create log dir and copy the config file
    basedir = args.basedir
    
    # args.expname = args.expname+'_mpi'
    expname=args.expname

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
            

    num_pt=args.num_pt
    kernel_window=args.kernel_window
    kernelnet = DSKnet(len(images), torch.tensor(poses[:, :3, :4]),
                           num_pt=num_pt, kernel_hwindow=kernel_window,
                           random_hwindow=0.15, in_embed=2,
                           random_mode='input',
                           img_embed=32,
                           spatial_embed=2,
                           depth_embed=0,
                           num_hidden=4,
                           num_wide=64,
                           short_cut=True,
                           optim_spatialvariant_trans=True)

    # Create nerf model
    args.N_image = images.shape[0]//2
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer,kernelnet = create_nerf(args,kernelnet)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }

    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    


    if args.render_bt:
        print('RENDER VIEW INTERPOLATION')
        
        render_poses = torch.Tensor(render_poses).to(device)
        print('target_idx ', target_idx)

        num_img = float(poses.shape[0])
        img_idx_embed = target_idx/float(num_img) * 2. - 1.0

        testsavedir = os.path.join(basedir, expname, 
                                'render-spiral-frame-%03d'%\
                                target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start))
        os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            render_bullet_time(render_poses, img_idx_embed, num_img, hwf, 
                               args.chunk, render_kwargs_test, 
                               gt_imgs=images, savedir=testsavedir, 
                               render_factor=args.render_factor)

        return

    if args.render_lockcam_slowmo:
        print('RENDER TIME INTERPOLATION')

        num_img = float(poses.shape[0])
        ref_c2w = torch.Tensor(ref_c2w).to(device)
        print('target_idx ', target_idx)

        testsavedir = os.path.join(basedir, expname, 'render-lockcam-slowmo')
        os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            render_lockcam_slowmo(ref_c2w, num_img, hwf, 
                            args.chunk, render_kwargs_test, 
                            gt_imgs=images, savedir=testsavedir, 
                            render_factor=args.render_factor,
                            target_idx=target_idx)

            return 

    if args.render_slowmo_bt:
        print('RENDER SLOW MOTION')

        curr_ts = 0
        render_poses = poses #torch.Tensor(poses).to(device)
        bt_poses = create_bt_poses(hwf) 
        bt_poses = bt_poses * 10

        with torch.no_grad():

            testsavedir = os.path.join(basedir, expname, 
                                    'render-slowmo_bt_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            images = torch.Tensor(images)#.to(device)

            print('render poses shape', render_poses.shape)
            render_slowmo_bt(depths, render_poses, bt_poses, 
                            hwf, args.chunk, render_kwargs_test,
                            gt_imgs=images, savedir=testsavedir, 
                            render_factor=args.render_factor, 
                            target_idx=10)
            # print('Done rendering', i,testsavedir)

        return

    if args.render_test:
        images = torch.Tensor(images)#.to(device)
        depths = torch.Tensor(depths)#.to(device)
        masks = torch.Tensor(bg_masks).to(device)
        num_img = float(images.shape[0])

        poses = torch.Tensor(poses).to(device)
        for img_i in i_test:
                target = images[img_i]
                pose = poses[img_i, :3,:4]

                img_idx_embed = img_i/num_img * 2. - 1.0
                testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(start))
                os.makedirs(testsavedir, exist_ok=True)
                with torch.no_grad():
                    # if i<args.N_iters:
                    ret = render(img_idx_embed,
                            0, False,
                            num_img, H, W, focal, 
                            chunk=1024*16, 
                            c2w=pose,
                            **render_kwargs_test)
                rgb = ret['rgb_map_ref'].cpu().numpy()
                save_img_path = os.path.join(testsavedir, 
                                    '%06d_right.png'%((img_i-1)//2))

                imageio.imwrite(save_img_path, 
                                to8b(rgb))
        return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    # Move training data to GPU
    images = torch.Tensor(images)#.to(device)
    depths = torch.Tensor(depths)#.to(device)
    masks = torch.Tensor(bg_masks).to(device)

    poses = torch.Tensor(poses).to(device)

    N_iters = args.N_iters+args.bokeh_iters+1 

    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    uv_grid = create_meshgrid(H, W, normalized_coordinates=False)[0].cuda() # (H, W, 2)

    # Summary writers
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    num_img = float(images.shape[0])
    
    decay_iteration = max(args.decay_iteration, 
                          args.end_frame - args.start_frame)
    decay_iteration = min(decay_iteration, 250)

    chain_bwd = 0



    for i in range(start, N_iters):
        chain_bwd = 1 - chain_bwd
        time0 = time.time()
        print('expname ', expname, ' chain_bwd ', chain_bwd, 
             ' lindisp ', args.lindisp, ' decay_iteration ', decay_iteration)
        print('Random FROM SINGLE IMAGE')
        # Random from one image
        img_i = np.random.choice(i_train)

        if i % (decay_iteration * 1000) == 0:
            torch.cuda.empty_cache()

        target = images[img_i].cuda()
        pose = poses[img_i, :3,:4]
        depth_gt = depths[img_i].cuda()

        hard_coords = torch.Tensor(motion_coords[img_i]).cuda()
        mask_gt = masks[img_i].cuda()


        index=img_i//2

        if index == 0:
            index=img_i//2
            flow_fwd, fwd_mask = read_optical_flow_NVDS(args.datadir, index, 
                                                args.start_frame, fwd=True)
            flow_bwd, bwd_mask = np.zeros_like(flow_fwd), np.zeros_like(fwd_mask)
        elif index == num_img//2 - 1:
            index=img_i//2
            flow_bwd, bwd_mask = read_optical_flow_NVDS(args.datadir, index, 
                                                args.start_frame, fwd=False)
            flow_fwd, fwd_mask = np.zeros_like(flow_bwd), np.zeros_like(bwd_mask)
        else:
            index=img_i//2
            flow_fwd, fwd_mask = read_optical_flow_NVDS(args.datadir, 
                                                index, args.start_frame, 
                                                fwd=True)
            flow_bwd, bwd_mask = read_optical_flow_NVDS(args.datadir, 
                                                index, args.start_frame, 
                                                fwd=False)

    
        flow_fwd = torch.Tensor(flow_fwd).cuda()
        fwd_mask = torch.Tensor(fwd_mask).cuda()
    
        flow_bwd = torch.Tensor(flow_bwd).cuda()
        bwd_mask = torch.Tensor(bwd_mask).cuda()
        # more correct way for flow loss
        flow_fwd = flow_fwd + uv_grid
        flow_bwd = flow_bwd + uv_grid

        if N_rand is not None:
            rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
            
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)


            if args.use_motion_mask and i <= decay_iteration * 1000:
                print('HARD MINING STAGE !')
                num_extra_sample = args.num_extra_sample
                print('num_extra_sample ', num_extra_sample)
                select_inds_hard = np.random.choice(hard_coords.shape[0], 
                                                    size=[min(hard_coords.shape[0], 
                                                        num_extra_sample)], 
                                                    replace=False)  # (N_rand,)
                select_inds_all = np.random.choice(coords.shape[0], 
                                                size=[N_rand], 
                                                replace=False)  # (N_rand,)

                select_coords_hard = hard_coords[select_inds_hard].long()
                select_coords_all = coords[select_inds_all].long()

                select_coords = torch.cat([select_coords_all, select_coords_hard], 0)
                rays_o = rays_o[select_coords[:, 0], 
                            select_coords[:, 1]]  
                rays_d = rays_d[select_coords[:, 0], 
                                select_coords[:, 1]]  
                batch_rays = torch.stack([rays_o, rays_d], 0) 
                target_rgb = target[select_coords[:, 0], 
                                    select_coords[:, 1]]  
                target_depth = depth_gt[select_coords[:, 0], 
                                    select_coords[:, 1]]
                target_mask= mask_gt[select_coords[:, 0], 
                                    select_coords[:, 1]]

                target_of_fwd = flow_fwd[select_coords[:, 0], 
                                        select_coords[:, 1]]
                target_fwd_mask = fwd_mask[select_coords[:, 0], 
                                        select_coords[:, 1]].unsqueeze(-1)

                target_of_bwd = flow_bwd[select_coords[:, 0], 
                                        select_coords[:, 1]]
                target_bwd_mask = bwd_mask[select_coords[:, 0], 
                                        select_coords[:, 1]].unsqueeze(-1)

            else:
                if i>2*decay_iteration*1000:
                    N_rand=384
                else:
                    N_rand=512
                select_inds = np.random.choice(coords.shape[0], 
                                            size=[N_rand], 
                                            replace=False)  
                select_coords = coords[select_inds].long()  
                xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
                xs = np.tile((xs[...,None] + 0.5), [1, 1, 1])
                ys = np.tile((ys[...,None] + 0.5), [1, 1, 1])
                xs=np.reshape(xs,[-1,1])
                ys=np.reshape(ys,[-1,1])
                rays_x=xs[select_inds]
                rays_y=ys[select_inds]
                rays_x=torch.Tensor(rays_x).cuda()
                rays_y=torch.Tensor(rays_y).cuda()
                rays_o = rays_o[select_coords[:, 0], 
                                select_coords[:, 1]] 
                rays_d = rays_d[select_coords[:, 0], 
                                select_coords[:, 1]]  
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_rgb = target[select_coords[:, 0], 
                                    select_coords[:, 1]]  
                target_depth = depth_gt[select_coords[:, 0], 
                                    select_coords[:, 1]]
                target_mask= mask_gt[select_coords[:, 0], 
                                    select_coords[:, 1]]
                target_of_fwd = flow_fwd[select_coords[:, 0], 
                                        select_coords[:, 1]]
                target_fwd_mask = fwd_mask[select_coords[:, 0], 
                                        select_coords[:, 1]].unsqueeze(-1)

                target_of_bwd = flow_bwd[select_coords[:, 0], 
                                        select_coords[:, 1]]
                target_bwd_mask = bwd_mask[select_coords[:, 0], 
                                        select_coords[:, 1]].unsqueeze(-1)
                
            

        img_idx_embed = img_i/num_img * 2. - 1.0

        #####  Core optimization loop  #####
        if args.chain_sf and i > decay_iteration * 1000 * 2:
            chain_5frames = True
        else:
            chain_5frames = False

        print('chain_5frames ', chain_5frames, ' chain_bwd ', chain_bwd)


        
        if i <= decay_iteration * 1000: 
            ret = render(img_idx_embed, 
                     chain_bwd, 
                     chain_5frames,
                     num_img, H, W, focal, 
                     chunk=args.chunk, 
                     rays=batch_rays,
                     verbose=i < 10, retraw=True,
                     **render_kwargs_train)
    
        else:
            new_rays, weight,align_loss = kernelnet(H, W, K, batch_rays,img_idx_embed,index, rays_x,rays_y)
            ray_num, pt_num = new_rays.shape[1:3]
            ret_rgb= render_mpi_dsk(img_idx_embed, 
                chain_bwd, 
                chain_5frames,
                num_img, H, W, focal, 
                chunk=args.chunk, 
                rays=new_rays.reshape(2,-1,3),
                verbose=i < 10, retraw=True,weight=weight,num_pt=num_pt,
                **render_kwargs_train)
            ret= render(img_idx_embed, 
                chain_bwd, 
                chain_5frames,
                num_img, H, W, focal, 
                chunk=args.chunk, 
                rays=batch_rays,
                verbose=i < 10, retraw=True,
                **render_kwargs_train)

        

        pose_post = poses[min(img_i + 1, int(num_img) - 1), :3,:4]
        pose_prev = poses[max(img_i - 1, 0), :3,:4]

        render_of_fwd, render_of_bwd = compute_optical_flow(pose_post, 
                                                            pose, pose_prev, 
                                                            H, W, focal, 
                                                            ret)

        optimizer.zero_grad()

        weight_map_post = ret['prob_map_post']
        weight_map_prev = ret['prob_map_prev']

        weight_post = 1. - ret['raw_prob_ref2post']
        weight_prev = 1. - ret['raw_prob_ref2prev']
        prob_reg_loss = args.w_prob_reg * (torch.mean(torch.abs(ret['raw_prob_ref2prev'])) \
                                + torch.mean(torch.abs(ret['raw_prob_ref2post'])))


        if i <= decay_iteration * 1000:
            rgb_dy=ret['rgb_map_ref_dy']
            rgb_post_dy=ret['rgb_map_post_dy']
            rgb_prev_dy=ret['rgb_map_prev_dy']
            rgb_ref=ret['rgb_map_ref']
            if chain_5frames:
                rgb_map_pp_dy=ret['rgb_map_pp_dy']
        else:

            rgb_dy=ret_rgb['bokeh_map_ref_dy']
            rgb_post_dy=ret_rgb['bokeh_map_post_dy']
            rgb_prev_dy=ret_rgb['bokeh_map_prev_dy']
            rgb_ref=ret_rgb['bokeh_map_ref']
            if chain_5frames:
                rgb_map_pp_dy=ret_rgb['bokeh_map_pp_dy']


        # dynamic rendering loss
        if i <= decay_iteration * 1000:

            render_loss = img2mse(rgb_dy, target_rgb)
            render_loss += compute_mse(rgb_post_dy, 
                                       target_rgb, 
                                       weight_map_post.unsqueeze(-1))
            render_loss += compute_mse(rgb_prev_dy, 
                                       target_rgb, 
                                       weight_map_prev.unsqueeze(-1))
        else:
            print('only compute dynamic render loss in masked region')
            weights_map_dd = ret['weights_map_dd'].unsqueeze(-1).detach()
            render_loss = compute_mse(rgb_dy, 
                                      target_rgb, 
                                      weights_map_dd)
            render_loss += compute_mse(rgb_post_dy, 
                                       target_rgb, 
                                       weight_map_post.unsqueeze(-1) * weights_map_dd)
            render_loss += compute_mse(rgb_prev_dy, 
                                       target_rgb, 
                                       weight_map_prev.unsqueeze(-1) * weights_map_dd)

            
        target_mask=target_mask+1e-5

        render_loss += img2mse(rgb_ref[:N_rand, ...], 
                            target_rgb[:N_rand, ...])
        if i<N_iters*0.8:
            if i >decay_iteration * 1000:
                render_loss += 0.1*align_loss

        sf_cycle_loss = args.w_cycle * compute_mae(ret['raw_sf_ref2post'], 
                                                   -ret['raw_sf_post2ref'], 
                                                   weight_post.unsqueeze(-1), dim=3) 
        sf_cycle_loss += args.w_cycle * compute_mae(ret['raw_sf_ref2prev'], 
                                                    -ret['raw_sf_prev2ref'], 
                                                    weight_prev.unsqueeze(-1), dim=3)
        
        # regularization loss

        render_sf_ref2prev = torch.sum(ret['weights_ref_dy'].unsqueeze(-1) * ret['raw_sf_ref2prev'], -1)
        render_sf_ref2post = torch.sum(ret['weights_ref_dy'].unsqueeze(-1) * ret['raw_sf_ref2post'], -1)
        
        sf_reg_loss = args.w_sf_reg * (torch.mean(torch.abs(render_sf_ref2prev)) \
                                    + torch.mean(torch.abs(render_sf_ref2post))) 
        

        divsor = i // (decay_iteration * 1000)

        decay_rate = 10

        if args.decay_depth_w:
            w_depth = args.w_depth/(decay_rate ** divsor)
        else:
            w_depth = args.w_depth

        if args.decay_optical_flow_w:
            w_of = args.w_optical_flow/(decay_rate ** divsor)
        else:
            w_of = args.w_optical_flow


        depth_loss = w_depth * (compute_depth_loss(ret['depth_map_ref_dy'], -target_depth))

        print('w_depth ', w_depth, 'w_of ', w_of)

        if index == 0:
            print('only fwd flow')
            flow_loss = w_of * compute_mae(render_of_fwd, 
                                        target_of_fwd, 
                                        target_fwd_mask)#torch.sum(torch.abs(render_of_fwd - target_of_fwd) * target_fwd_mask)/(torch.sum(target_fwd_mask) + 1e-8)
        elif index == num_img//2 - 1:
            print('only bwd flow')
            flow_loss = w_of * compute_mae(render_of_bwd, 
                                        target_of_bwd, 
                                        target_bwd_mask)#torch.sum(torch.abs(render_of_bwd - target_of_bwd) * target_bwd_mask)/(torch.sum(target_bwd_mask) + 1e-8)
        else:
            flow_loss = w_of * compute_mae(render_of_fwd, 
                                        target_of_fwd, 
                                        target_fwd_mask)#torch.sum(torch.abs(render_of_fwd - target_of_fwd) * target_fwd_mask)/(torch.sum(target_fwd_mask) + 1e-8)
            flow_loss += w_of * compute_mae(render_of_bwd, 
                                        target_of_bwd, 
                                        target_bwd_mask)#torch.sum(torch.abs(render_of_bwd - target_of_bwd) * target_bwd_mask)/(torch.sum(target_bwd_mask) + 1e-8)

        # scene flow smoothness loss
        sf_sm_loss = args.w_sm * (compute_sf_sm_loss(ret['raw_pts_ref'], 
                                                    ret['raw_pts_post'], 
                                                    H, W, focal) \
                                + compute_sf_sm_loss(ret['raw_pts_ref'], 
                                                    ret['raw_pts_prev'], 
                                                    H, W, focal))

        # scene flow least kinectic loss
        sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_ref'], 
                                                    ret['raw_pts_post'], 
                                                    ret['raw_pts_prev'], 
                                                    H, W, focal)
        sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_ref'], 
                                                    ret['raw_pts_post'], 
                                                    ret['raw_pts_prev'], 
                                                    H, W, focal)
        entropy_loss = args.w_entropy * torch.mean(-ret['raw_blend_w'] * torch.log(ret['raw_blend_w'] + 1e-8))

        # # ======================================  two-frames chain loss ===============================
        if chain_bwd:
            sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_prev'], 
                                                          ret['raw_pts_ref'], 
                                                          ret['raw_pts_pp'], 
                                                          H, W, focal)

        else:
            sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_post'], 
                                                          ret['raw_pts_pp'], 
                                                          ret['raw_pts_ref'], 
                                                          H, W, focal)

        if chain_5frames:
            render_loss += compute_mse(rgb_map_pp_dy, 
                                target_rgb, 
                                weights_map_dd)


        loss = sf_reg_loss + sf_cycle_loss + \
               render_loss + flow_loss + \
               sf_sm_loss + prob_reg_loss + \
               depth_loss + entropy_loss 

        print('render_loss ', render_loss.item(), 
              ' bidirection_loss ', sf_cycle_loss.item(), 
              ' sf_reg_loss ', sf_reg_loss.item())
        print('depth_loss ', depth_loss.item(), 
              ' flow_loss ', flow_loss.item(), 
              ' sf_sm_loss ', sf_sm_loss.item())
        print('prob_reg_loss ', prob_reg_loss.item(),
              ' entropy_loss ', entropy_loss.item())
        # with torch.autograd.detect_anomaly():
        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))

            if args.N_importance > 0:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_rigid': render_kwargs_train['network_rigid'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'kernel_state_dict': kernelnet.state_dict(),
                }, path)
            
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_rigid': render_kwargs_train['network_rigid'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'kernel_state_dict': kernelnet.state_dict(),
                }, path)

            print('Saved checkpoints at', path)


        if i % args.i_print == 0 and i > 0:
            writer.add_scalar("train/loss", loss.item(), i)
            
            writer.add_scalar("train/render_loss", render_loss.item(), i)
            writer.add_scalar("train/depth_loss", depth_loss.item(), i)
            writer.add_scalar("train/flow_loss", flow_loss.item(), i)
            writer.add_scalar("train/prob_reg_loss", prob_reg_loss.item(), i)

            writer.add_scalar("train/sf_reg_loss", sf_reg_loss.item(), i)
            writer.add_scalar("train/sf_cycle_loss", sf_cycle_loss.item(), i)
            writer.add_scalar("train/sf_sm_loss", sf_sm_loss.item(), i)


        if i%args.i_img == 0:
            target = images[img_i]
            pose = poses[img_i, :3,:4]
            target_depth = depths[img_i] - torch.min(depths[img_i])
            target_mask= masks[img_i] - torch.min(masks[img_i])

            img_idx_embed = img_i/num_img * 2. - 1.0
            index=img_i//2
            if index == 0:
                
                flow_fwd, fwd_mask = read_optical_flow_NVDS(args.datadir, index, 
                                                       args.start_frame, fwd=True)
                flow_bwd, bwd_mask = np.zeros_like(flow_fwd), np.zeros_like(fwd_mask)

            elif index == num_img//2 - 1:
                flow_bwd, bwd_mask = read_optical_flow_NVDS(args.datadir, index, 
                                                       args.start_frame, fwd=False)
                flow_fwd, fwd_mask = np.zeros_like(flow_bwd), np.zeros_like(bwd_mask)
            else:
                flow_fwd, fwd_mask = read_optical_flow_NVDS(args.datadir, 
                                                       index, args.start_frame, 
                                                       fwd=True)
                flow_bwd, bwd_mask = read_optical_flow_NVDS(args.datadir, 
                                                       index, args.start_frame, 
                                                       fwd=False)

            flow_fwd_rgb = torch.Tensor(flow_to_image(flow_fwd)/255.)#.cuda()
            writer.add_image("val/gt_flow_fwd", 
                            flow_fwd_rgb, global_step=i, dataformats='HWC')
            flow_bwd_rgb = torch.Tensor(flow_to_image(flow_bwd)/255.)#.cuda()
            writer.add_image("val/gt_flow_bwd", 
                            flow_bwd_rgb, global_step=i, dataformats='HWC')

            with torch.no_grad():
                # if i< decay_iteration*1000:
                ret = render(img_idx_embed,
                        chain_bwd, False,
                        num_img, H, W, focal, 
                        chunk=1024*16, 
                        c2w=pose,
                        **render_kwargs_test)
                
                writer.add_image("val/rgb_map_ref", torch.clamp(ret['rgb_map_ref'], 0., 1.), 
                                global_step=i, dataformats='HWC')

                writer.add_image("val/depth_map_ref", normalize_depth(ret['depth_map_ref']), 
                                global_step=i, dataformats='HW')

                writer.add_image("val/rgb_map_rig", torch.clamp(ret['rgb_map_rig'], 0., 1.), 
                                global_step=i, dataformats='HWC')
                
                writer.add_image("val/depth_map_rig", normalize_depth(ret['depth_map_rig']), 
                                global_step=i, dataformats='HW')

                writer.add_image("val/rgb_map_ref_dy", torch.clamp(ret['rgb_map_ref_dy'], 0., 1.), 
                                global_step=i, dataformats='HWC')
                
                writer.add_image("val/depth_map_ref_dy", normalize_depth(ret['depth_map_ref_dy']), 
                                global_step=i, dataformats='HW')

                writer.add_image("val/gt_rgb", target, 
                                global_step=i, dataformats='HWC')
                writer.add_image("val/monocular_disp", 
                                torch.clamp(target_depth /percentile(target_depth, 97), 0., 1.), 
                                global_step=i, dataformats='HW')
                writer.add_image("val/mask_rig", 
                                torch.clamp(target_mask /percentile(target_mask, 97), 0., 1.), 
                                global_step=i, dataformats='HW')

                writer.add_image("val/weights_map_dd", 
                                 ret['weights_map_dd'], 
                                 global_step=i, 
                                 dataformats='HW')

            # torch.cuda.empty_cache()

        if i%args.i_testset==0 and i > 0:
            print('test poses shape', poses[i_test].shape)
            for img_i in i_test:
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                target_depth = depths[img_i] - torch.min(depths[img_i])

                img_idx_embed = img_i/num_img * 2. - 1.0
                testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
                os.makedirs(testsavedir, exist_ok=True)
                with torch.no_grad():
                    ret = render(img_idx_embed,
                            chain_bwd, False,
                            num_img, H, W, focal, 
                            chunk=1024*16, 
                            c2w=pose,
                            **render_kwargs_test)
                    rgb = ret['rgb_map_ref'].cpu().numpy()

                    
                    save_img_path = os.path.join(testsavedir, 
                                        '%06d_right.png'%((img_i-1)//2))

                    imageio.imwrite(save_img_path, 
                                    to8b(rgb))
                

            
            
            print('Saved test set')

        
        
        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
