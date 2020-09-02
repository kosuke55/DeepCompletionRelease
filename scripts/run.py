import argparse
import os
import subprocess
from pathlib import Path


list_path = '../torch/data_list/realsense_list.txt'


def execute(cmd):
    print(cmd)
    subprocess.run(cmd, shell=True)


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
parser.add_argument(
    '--inertia-weight', '-iw', type=float,
    help='inertia weight',
    default=1000.)
parser.add_argument(
    '--smmothness-weight', '-sw', type=float,
    help='smmothness weight',
    default=0.001)
parser.add_argument(
    '--tangent-weight', '-tw', type=float,
    help='tangent weight',
    default=1.)

args = parser.parse_args()

idx = args.idx
color_path = args.color
depth_path = args.depth
camera_info_path = args.camera_info
output_dir = Path(args.output_dir)
os.makedirs(str(output_dir), exist_ok=True)
output_depth_path = str(output_dir / '{:03}.png'.format(idx))
inertia_weight = args.inertia_weight
smmothness_weight = args.smmothness_weight
tangent_weight = args.tangent_weight

with open(list_path, "w") as f:
    f.write('{:03}'.format(idx))

# Prepare data
execute('python prepare_data.py -i {} -c {} -d {} -ci {} -o {}'.format(
    idx, color_path, depth_path, camera_info_path, output_dir
))

# DeepCompletion
execute('sudo docker run --gpus all --rm -it -v /home/kosuke55/DeepCompletionRelease:/root/dc deep_completion /bin/bash -i -c "cd /root/dc/torch; th main_test_bound_realsense.lua -test_model ../pre_train_model/bound.t7 -test_file ./data_list/realsense_list.txt -root_path ../data/realsense/"')
execute('sudo docker run --gpus all --rm -it -v /home/kosuke55/DeepCompletionRelease:/root/dc deep_completion /bin/bash -i -c "cd /root/dc/torch; th main_test_realsense.lua -test_model ../pre_train_model/normal_scannet.t7 -test_file ./data_list/realsense_list.txt -root_path ../data/realsense/"')
execute('python generate_occlusion_weight.py -i {}'.format(idx))
execute('python compose_depth.py -i {:03} -o {} -iw {} -sw {} -tw {}'.format(
    idx, output_dir, inertia_weight, smmothness_weight, tangent_weight))

# Create PointCloud
execute(
    'ipython -i -- create_pointcloud.py -d {} -c ../data/realsense/{:03}_color.png -ds 4000 -ci {}'.format(
        output_depth_path,
        idx,
        output_dir / 'camera_info/{:03}_camera_info'.format(idx)))
