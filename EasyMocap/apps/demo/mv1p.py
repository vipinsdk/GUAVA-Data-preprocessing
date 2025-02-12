'''
  @ Date: 2021-04-13 19:46:51
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-13 17:56:25
  @ FilePath: /EasyMocap/apps/demo/mv1p.py
'''
from tqdm import tqdm
from easymocap.smplmodel import check_keypoints, load_model, select_nf
from easymocap.mytools import simple_recon_person, Timer, projectN3
from easymocap.pipeline import smpl_from_keypoints3d2d
import os
import trimesh
from os.path import join
import numpy as np
import torch
import torchvision
from PIL import Image
import torchvision.transforms.functional as F_t
import torchvision.transforms as T
# from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_euler_angles
# from pytorch3d.utils import cameras_from_opencv_projection


def get_mask_tilted_line(verts, h=1920, w=1080): 
        # Adjusting input to handle batched input: assume verts_ndc has shape (B, V, 3)
        batch_size = verts.shape[0]
        verts = torch.Tensor(verts).to(device='cuda:0')
        # Scaling verts_xy to image space for each batch
        # verts_xy = (verts_ndc * 0.5 + 0.5) * torch.tensor([w, h], device=verts_ndc.device)
        verts_xy = verts[..., :2]  # Shape: (B, V, 2)
        # Get specific regions for each batch
        verts_xy_left = verts_xy[:, 6616]
        verts_xy_right = verts_xy[:, 5603]
        verts_xy_bottom = verts_xy[:, 4110]
 
        # Compute slope and intercept (k and b) for each item in the batch
        delta_xy = verts_xy_left - verts_xy_right
        assert (delta_xy[:, 0] != 0).all()
        k = delta_xy[:, 1] / delta_xy[:, 0]
        b = verts_xy_bottom[:, 1] - k * verts_xy_bottom[:, 0]

        # Prepare mesh grid for each batch
        x = torch.arange(w, device=verts.device)
        y = torch.arange(h, device=verts.device)
        yx = torch.stack(torch.meshgrid(y, x, indexing='ij'), dim=-1)  # Shape: (h, w, 2)
        
        # Expanding yx to match batch dimensions
        yx = yx[None, :, :, :]  # Shape: (1, h, w, 2)

        # Calculate mask for each batch item
        k = k[:, None, None]  # Expand k to (B, 1, 1)
        b = b[:, None, None]  # Expand b to (B, 1, 1)
        mask = ((k * yx[..., 1] + b - yx[..., 0]) > 0).float()  # Shape: (B, h, w)

        # Apply anti-aliasing with Gaussian kernel
        kernel_size = int(0.03 * w) // 2 * 2 + 1
        blur = torchvision.transforms.GaussianBlur(kernel_size, sigma=kernel_size)
        
        # Apply blur for each mask in the batch
        mask = torch.stack([blur(mask[i:i+1])[0] for i in range(batch_size)], dim=0)  # Shape: (B, h, w)
        return mask

def mask_image_with_mask(image_path, output_folder, mask, nf, nViews=15, device='cuda:0'):
    transform = T.ToPILImage()
    for j in range(nViews):
        rgb_path = image_path + "/" + f"{nf:05d}_{j:02d}.png"
        mask_path = rgb_path.replace("images", "fg_masks")
        image = F_t.to_tensor(Image.open(rgb_path).convert("RGBA")).to(device)
        fg_mask = F_t.to_tensor(Image.open(mask_path).convert("RGB")).to(device)

        # Create a white background (1, H, W) - in this case, white is [1, 1, 1] for RGB
        white_background = torch.ones_like(image[:3, :, :], device=device)  # RGB channels only
        # Set the alpha channel
        alpha = image[3, :, :]  # Alpha channel (transparency)
        # Replace areas where alpha is 0 (transparent) with the white background
        image[:3, :, :] = alpha.unsqueeze(0) * image[:3, :, :] + (1 - alpha.unsqueeze(0)) * white_background
        # Now image is RGB, with a white background in transparent regions
        image = image[:3, :, :]  # Drop alpha channel, keeping only RGB

        mask_float = mask[j].float()
        image = image * mask_float[None, :, :] + (1 - mask_float[None, :, :]) * white_background
        fg_mask = mask[j] * fg_mask 
        img_save_path = f"{output_folder}" + "/images/" + f"{nf:05d}_{j:02d}.png"
        mask_save_path = img_save_path.replace("images", "fg_masks")
        if not os.path.exists(os.path.dirname(img_save_path)):
            os.makedirs(os.path.dirname(img_save_path))
            os.makedirs(os.path.dirname(mask_save_path))
        img = transform(image.cpu())
        fg_mask = transform(fg_mask.cpu())
        img.save(img_save_path)
        fg_mask.save(mask_save_path)

def check_repro_error(keypoints3d, kpts_repro, keypoints2d, P, MAX_REPRO_ERROR):
    square_diff = (keypoints2d[:, :, :2] - kpts_repro[:, :, :2])**2 
    conf = keypoints3d[None, :, -1:]
    conf = (keypoints3d[None, :, -1:] > 0) * (keypoints2d[:, :, -1:] > 0)
    dist = np.sqrt((((kpts_repro[..., :2] - keypoints2d[..., :2])*conf)**2).sum(axis=-1))
    vv, jj = np.where(dist > MAX_REPRO_ERROR)
    if vv.shape[0] > 0:
        keypoints2d[vv, jj, -1] = 0.
        keypoints3d, kpts_repro = simple_recon_person(keypoints2d, P)
    return keypoints3d, kpts_repro

def mv1pmf_skel(dataset, check_repro=True, args=None):
    MIN_CONF_THRES = args.thres2d
    no_img = not (args.vis_det or args.vis_repro)
    dataset.no_img = no_img
    kp3ds = []
    start, end = args.start, min(args.end, len(dataset))
    kpts_repro = None
    for nf in tqdm(range(start, end), desc='triangulation'):
        images, annots = dataset[nf]
        check_keypoints(annots['keypoints'], WEIGHT_DEBUFF=1, min_conf=MIN_CONF_THRES)
        # print(annots['keypoints'].shape)
        keypoints3d, kpts_repro = simple_recon_person(annots['keypoints'], dataset.Pall)
        if check_repro:
            keypoints3d, kpts_repro = check_repro_error(keypoints3d, kpts_repro, annots['keypoints'], P=dataset.Pall, MAX_REPRO_ERROR=args.MAX_REPRO_ERROR)
        # keypoints3d, kpts_repro = robust_triangulate(annots['keypoints'], dataset.Pall, config=config, ret_repro=True)
        kp3ds.append(keypoints3d)
        if args.vis_det:
            dataset.vis_detections(images, annots, nf, sub_vis=args.sub_vis)
        if args.vis_repro:
            dataset.vis_repro(images, kpts_repro, nf=nf, sub_vis=args.sub_vis)
    # smooth the skeleton
    if args.smooth3d > 0:
        kp3ds = smooth_skeleton(kp3ds, args.smooth3d)
    for nf in tqdm(range(len(kp3ds)), desc='dump'):
        dataset.write_keypoints3d(kp3ds[nf], nf+start)

def mv1pmf_smpl(dataset, args, weight_pose=None, weight_shape=None):
    dataset.skel_path = args.skel
    kp3ds = []
    start, end = args.start, min(args.end, len(dataset))
    keypoints2d, bboxes, params_out = [], [], []
    mesh_translation = []
    dataset.no_img = True
    for nf in tqdm(range(start, end), desc='loading'):
        images, annots = dataset[nf]
        keypoints2d.append(annots['keypoints'])
        bboxes.append(annots['bbox'])
    kp3ds = dataset.read_skeleton(start, end)
    keypoints2d = np.stack(keypoints2d)
    bboxes = np.stack(bboxes)
    kp3ds = check_keypoints(kp3ds, 1)
    # optimize the human shape
    with Timer('Loading {}, {}'.format(args.model, args.gender), not args.verbose):
        body_model = load_model(gender=args.gender, model_type=args.model)
    params = smpl_from_keypoints3d2d(body_model, kp3ds, keypoints2d, bboxes, 
        dataset.Pall, config=dataset.config, args=args,
        weight_shape=weight_shape, weight_pose=weight_pose)
    # write out the results
    dataset.no_img = not (args.vis_smpl or args.vis_repro)
    for nf in tqdm(range(start, end), desc='render'):
        images, annots = dataset[nf]
        param = select_nf(params, nf-start)

        if args.flame_path is not None:
            flame_param_path = os.path.join(args.flame_path, '{:05d}.npz'.format(nf))
            flame_param = dict(np.load(flame_param_path))
            for key in flame_param.keys():
                flame_param[key] = torch.from_numpy(flame_param[key])

            mano_param_path = os.path.join(args.mano_path, '8_{:06d}.npz'.format(nf))
            mano_params = np.load(mano_param_path, allow_pickle=True)
            rmano_params = mano_params['righthand'].tolist()
            lmano_params = mano_params['lefthand'].tolist()
            for key in rmano_params.keys():
                rmano_params[key] = torch.from_numpy(rmano_params[key])

            # for key in lmano_params.keys():
            #     lmano_params[key] = torch.from_numpy(lmano_params[key])

            param['jaw_pose'] = matrix_to_euler_angles(rotation_6d_to_matrix(flame_param['jaw_pose']), convention='XYZ')
            param['reye_pose'] = matrix_to_euler_angles(rotation_6d_to_matrix(flame_param['eyes_pose'][:,:6]), convention='XYZ')
            param['leye_pose'] = matrix_to_euler_angles(rotation_6d_to_matrix(flame_param['eyes_pose'][:,6:]), convention='XYZ')
            param['expression'] = flame_param['expr']
            param['right_hand_pose'] = matrix_to_euler_angles(rmano_params['hand_pose'], convention='XYZ').unsqueeze(0)
            param['left_hand_pose'] = matrix_to_euler_angles(lmano_params['hand_pose'], convention='XYZ').unsqueeze(0)
        
        params_out.append(param)
        mesh_translation.append(param['transl'])
        smplx_params_path = join(args.out, 'smplx_params')
        os.makedirs(smplx_params_path, exist_ok=True)
        np.savez(join(smplx_params_path ,f'{nf:05d}.npz'), **param)
        # dataset.write_smpl(param, nf)

        if args.write_smpl_full:
            param_full = param.copy()
            param_full['poses'] = body_model.full_poses(param['poses'])
            dataset.write_smpl(param_full, nf, mode='smpl_full')
        if args.write_vertices:
            vertices = body_model(return_verts=True, return_tensor=False, **param)
            write_data = [{'id': 0, 'vertices': vertices[0]}]
            mesh_folder = join(args.out, 'mesh')
            # dataset.write_vertices(write_data, nf)
            os.makedirs(mesh_folder, exist_ok=True)
            trimesh.Trimesh(faces=body_model.faces, vertices=vertices[0], process=False).export(f'{mesh_folder}/{nf:06d}.obj')
        if args.vis_smpl: 
            vertices = body_model(return_verts=True, return_tensor=False, **param)
            vertices_ndc = projectN3(vertices[0], Pall=dataset.Pall)
            mask = get_mask_tilted_line(vertices_ndc[:, :, :2])
            mask_image_with_mask('/netscratch/jeetmal/output/BiRefNet/Ameer_full_setup/images', args.out, mask, nf, nViews=len(dataset.Pall))
            dataset.vis_smpl(vertices=vertices[0], faces=body_model.faces, images=images, nf=nf, sub_vis=args.sub_vis, add_back=True)
        if args.vis_repro:
            keypoints = body_model(return_verts=False, return_tensor=False, **param)[0]
            kpts_repro = projectN3(keypoints, dataset.Pall)
            dataset.vis_repro(images, kpts_repro, nf=nf, sub_vis=args.sub_vis, mode='repro_smpl')

    np.savez(join(args.out, 'params.npz'), **{'params': params_out})

def write_canonical_flame_param(params, tgt_folder):
    smplx_param = {
        'transl': np.zeros_like(params['transl'][:1]),
        'global_orient': np.zeros_like(params['global_orient'][:1]),
        'jaw_pose': np.array([[0.3, 0, 0]]),  # open mouth
        'lyes_pose': np.zeros_like(params['leye_pose'][:1]),
        'reye_pose': np.zeros_like(params['reye_pose'][:1]),
        'shape': params['shapes'][:1],
        'expression': np.zeros_like(params['expression'][:1]),
        'left_hand_pose': np.zeros_like(params['left_hand_pose'][:1]),
        'right_hand_pose': np.zeros_like(params['right_hand_pose'][:1]),
        'body_pose': np.zeros_like(params['body_pose'][:1]),
    }
    
    cano_smplx_param_path = tgt_folder / 'canonical_smplx_param.npz'
    print(f"Writing canonical FLAME parameters to: {cano_smplx_param_path}")
    np.savez(cano_smplx_param_path, **smplx_param)

def mv1pmf_output(dataset, args):
    params = np.load(join(args.out, 'params.npz'), allow_pickle=True)['params']
    write_canonical_flame_param(params, args.out)

    # for nf, item in tqdm(enumerate(params), total=len(params), desc='smplx_params'):
    #     # item['transl'] = (M[:3, 3] + item['transl']).numpy()
    #     smplx_params_path = join(args.out, 'smplx_params_new')
    #     os.makedirs(smplx_params_path, exist_ok=True)
    #     for key, val in item.items():
    #         if isinstance(val, torch.Tensor):
    #             item[key] = val.numpy()
    #     np.savez(join(smplx_params_path ,f'{nf:05d}.npz'), **item)
    dataset.write()
    
if __name__ == "__main__":
    from easymocap.mytools import load_parser, parse_parser
    from easymocap.dataset import CONFIG, MV1PMF
    parser = load_parser()
    parser.add_argument('--skel', action='store_true')
    parser.add_argument('--mesh_root', type=str, default=None)
    parser.add_argument('--flame_path', type=str, default=None)
    parser.add_argument('--mano_path', type=str, default=None)
    args = parse_parser(parser)
    help="""
  Demo code for multiple views and one person:

    - Input : {} => {}
    - Output: {}
    - Body  : {}=>{}, {}
""".format(args.path, ', '.join(args.sub), args.out, 
    args.model, args.gender, args.body)
    print(help)
    skel_path = join(args.out, 'keypoints3d')
    dataset = MV1PMF(args.path, annot_root=args.annot, cams=args.sub, out=args.out,
        config=CONFIG[args.body], kpts_type=args.body,
        undis=args.undis, no_img=False, verbose=args.verbose, mano_path=args.mano_path)
    dataset.writer.save_origin = args.save_origin

    if args.skel or not os.path.exists(skel_path):
         mv1pmf_skel(dataset, check_repro=True, args=args)
    # mv1pmf_smpl(dataset, args)
    mv1pmf_output(dataset, args)
    