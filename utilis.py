import os
import torch
import numpy as np
import json


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['test']
    metas = {}
    with open(basedir, 'r') as fp:
        metas['test'] = json.load(fp) # 'transforms_{}.json'
    basedir = os.path.dirname(basedir)

    file_name = []
    all_poses = []
    #for s in splits:
    meta = metas['test']
    poses = []
    if testskip==0:#s=='train' or 
        skip = 1
    else:
        skip = testskip

    for frame in meta['frames'][::skip]:
        file_name.append(frame['file_path'])
        poses.append(np.array(frame['transform_matrix']))
    poses = np.array(poses).astype(np.float32)
    all_poses.append(poses)
    poses = np.concatenate(all_poses, 0)

    H = 800 
    W = 800
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

    return poses, render_poses, [H, W, focal], file_name


def load_data(args):
    K, depths = None, None
    near_clip = None

    if args.dataset_type == 'blender':
        poses, render_poses, hwf, file_name = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', render_poses.shape, hwf, args.datadir)
        near, far = 2., 6.

    else:
        raise NotImplementedError(f'Unknown dataset type {args.dataset_type} exiting')

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([[800, 800] for i in range(poses.shape[0])])

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[...,:4]

    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks,
        near=near, far=far,
        poses=poses, render_poses=render_poses,
        file_name=file_name,
    )
    return data_dict


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()  # could be too small to exist the scene bbox
                      # it is only used to determined scene bbox
                      # lib/dvgo use 1e9 as far
    near = far * ratio
    return near, far
