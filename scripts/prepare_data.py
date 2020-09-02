import argparse
import os

import cv2
import numpy as np
from pathlib import Path
from cameramodels import PinholeCameraModel

data_path = Path('../data/realsense')

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--idx', '-i', type=int,
    help='input urdf',
    required=True)
parser.add_argument(
    '--color', '-c', type=str,
    help='color image',
    required=True)
parser.add_argument(
    '--depth', '-d', type=str,
    help='depth image',
    required=True)
parser.add_argument(
    '--camera-info', '-ci', type=str,
    help='camera_info',
    required=True)
parser.add_argument(
    '--output-dir', '-o', type=str,
    help='output dir',
    default='../results/kosuke55')

args = parser.parse_args()

# color_path = '502_color.png'
# depth_path = '502_depth_open.npy'
idx = args.idx
color_path = args.color
depth_path = args.depth
camera_info_path = args.camera_info
output_dir = Path(args.output_dir)

output_color_path = str(data_path / '{:03}_color.png'.format(idx))
output_depth_path = str(data_path / '{:03}_depth_open.png'.format(idx))
os.makedirs(str(output_dir / 'camera_info'), exist_ok=True)
output_camera_info_path = str(output_dir / 'camera_info/{:03}_camera_info'.format(idx))

# depth_png_path = str(Path(depth_path).with_suffix('.png'))
# camera_info_path = '502_camera_info.yaml'

target_size = (320, 240)

color = cv2.imread(color_path)
resized_color = cv2.resize(color, target_size)
cv2.imwrite(output_color_path, resized_color)

depth = np.load(depth_path)
resized_depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)
rescale_depth = (resized_depth / 1000. * 4000).astype(np.uint16)
cv2.imwrite(output_depth_path, rescale_depth)

cm = PinholeCameraModel.from_yaml_file(camera_info_path)
cm.target_size = target_size
cm.dump(output_camera_info_path)
