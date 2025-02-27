# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de

import argparse
from pathlib import Path

from yacs.config import CfgNode as CN

cfg = CN()

cfg.mt_dir = '/netscratch/jeetmal/models/metrical-tracker/'
cfg.flame_geom_path = cfg.mt_dir + 'data/FLAME2020/male_model.pkl'
# cfg.flame_geom_path = cfg.mt_dir + 'data/flame2023.pkl'
cfg.flame_template_path = cfg.mt_dir + 'data/uv_template.obj'
cfg.flame_lmk_path = cfg.mt_dir + 'data/landmark_embedding.npy'
# cfg.flame_lmk_path = cfg.mt_dir + 'data/landmark_embedding_with_eyes.npy'
cfg.tex_space_path = cfg.mt_dir + 'data/FLAME2020/FLAME_texture.npz'

cfg.num_shape_params = 300
cfg.num_exp_params = 100
cfg.tex_params = 140
cfg.actor = ''
cfg.config_name = ''
cfg.camera_id = 0
cfg.kernel_size = 7
cfg.sigma = 9.0
cfg.keyframes = [0]
cfg.bbox_scale = 2.5
cfg.fps = 30
cfg.begin_frames = 0
cfg.end_frames = 0
cfg.image_size = [1920, 1080]  # height, width
# cfg.image_size = [512, 512]  # height, width
# cfg.rotation_lr = 0.2
# cfg.translation_lr = 0.003
cfg.rotation_lr = 0.05
cfg.translation_lr = 0.004
cfg.raster_update = 8
cfg.pyr_levels = [[1.0, 160], [0.25, 40], [0.5, 40], [1.0, 70]]  # Gaussian pyramid levels (scaling, iters per level) first level is only the sparse term!
cfg.optimize_shape = False
cfg.optimize_jaw = False
cfg.optimize_neck = False
cfg.undistort = False
cfg.crop_image = False
cfg.alpha_map = False
cfg.root_folder = '/netscratch/jeetmal/videos/capture_10_02_2025/Ameer/videos'
cfg.save_folder = ''
cfg.flame_param = False
cfg.flame_param_path = '/netscratch/jeetmal/output/metrical-tracker/'

# Weights
cfg.w_pho = 350
cfg.w_lmks = 7000
cfg.w_lmks_68 = 1000
cfg.w_lmks_lid = 1000
cfg.w_lmks_mouth = 15000
cfg.w_lmks_iris = 1000
cfg.w_lmks_oval = 2000

cfg.w_exp = 0.02
cfg.w_shape = 0.3
cfg.w_tex = 0.04
cfg.w_jaw = 0.05
cfg.w_neck = 0.05


def get_cfg_defaults():
    return cfg.clone()


def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Configuration file', required=True)

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg = get_cfg_defaults()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    cfg.config_name = Path(args.cfg).stem

    return cfg


def parse_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg = update_cfg(cfg, cfg_file)
    cfg.cfg_file = cfg_file

    cfg.config_name = Path(cfg_file).stem

    return cfg
