import argparse
import os.path as osp
import sys

import cameramodels
import open3d as o3d
import numpy as np
import skrobot
import trimesh

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--color', '-c', type=str,
    help='color image',
    default='')
parser.add_argument(
    '--depth', '-d', type=str,
    help='depth image',
    default='')
parser.add_argument(
    '--depth-scale', '-ds', type=float,
    help='depth scale', default=-1)
parser.add_argument(
    '--camera-info', '-ci', type=str,
    help='camera info',
    default='')

args = parser.parse_args()

color_path = args.color
depth_path = args.depth
depth_scale = args.depth_scale
camera_info_path = args.camera_info

color = o3d.io.read_image(color_path)
depth = o3d.io.read_image(depth_path)

if depth_scale > 0:
    depth = o3d.geometry.Image((np.asarray(depth) / depth_scale).astype(np.float32))

cameramodel = cameramodels.PinholeCameraModel.from_yaml_file(camera_info_path)
intrinsics = cameramodel.open3d_intrinsic

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd, intrinsics)
o3d.visualization.draw_geometries([pcd])    