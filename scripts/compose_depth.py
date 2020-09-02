import os.path as osp
import argparse
import subprocess

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--idx', '-i', type=int,
    help='input urdf',
    default='502'
)

args = parser.parse_args()
idx = args.idx

inertia_weight = 1000.
smmothness_weight = 0.001
tangent_weight = 1.

depth2depth_path = '../gaps/bin/x86_64/depth2depth'
normal_path = '../torch/result/normal_scannet_realsense_test'
normal_weight_path = '../torch/result/bound_realsense_weight'

input_depth_png = '../data/realsense/{:03}_depth_open.png'.format(idx)
output_depth_png = '../results/kosuke55/{:03}.png'.format(idx)
input_normal_h5 = '../torch/result/normal_scannet_realsense_test/realsense_{:03}_normal_est.h5'.format(idx)
normal_weight_png = osp.join(normal_weight_path, 'realsense_{:03}_weight.png'.format(idx))

cmd = '{} {} {} -xres {} -yres {} -fx {} -fy {} -cx {} -cy {} -inertia_weight {} -smoothness_weight {}'.format(
    depth2depth_path, input_depth_png, output_depth_png,
    320, 240, 308.331, 308.331, 165.7475, 119.8889,
    inertia_weight, smmothness_weight)

if osp.isdir(normal_path):
    cmd = '{} -tangent_weight {} -input_normals {} '.format(
        cmd, tangent_weight, input_normal_h5)

if osp.isdir(normal_weight_path):
    cmd = '{} -input_tangent_weight {}'.format(
        cmd, normal_weight_png
    )

print(cmd)
subprocess.run(cmd, shell=True)
