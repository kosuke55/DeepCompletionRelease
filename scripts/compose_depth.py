import os.path as osp
import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--idx', '-i', type=int,
    help='idx',
    required=True
)
parser.add_argument(
    '--output-dir', '-o', type=str,
    help='output dir',
    default='../results/kosuke55')
parser.add_argument(
    '--inertia-weight', '-iw', type=float,
    help='inertia weight',
    default=1000.
)
parser.add_argument(
    '--smmothness-weight', '-sw', type=float,
    help='smmothness weight',
    default=0.001
)
parser.add_argument(
    '--tangent-weight', '-tw', type=float,
    help='tangent weight',
    default=1.
)

args = parser.parse_args()
idx = args.idx
output_dir = Path(args.output_dir)

inertia_weight = args.inertia_weight
smmothness_weight = args.smmothness_weight
tangent_weight = args.tangent_weight

depth2depth_path = '../gaps/bin/x86_64/depth2depth'
normal_path = '../torch/result/normal_scannet_realsense_test'
normal_weight_path = '../torch/result/bound_realsense_weight'

input_depth_png = '../data/realsense/{:03}_depth_open.png'.format(idx)
output_depth_path = str(output_dir / '{:03}.png'.format(idx))
input_normal_h5 = '../torch/result/normal_scannet_realsense_test/realsense_{:03}_normal_est.h5'.format(idx)
normal_weight_png = osp.join(normal_weight_path, 'realsense_{:03}_weight.png'.format(idx))

cmd = '{} {} {} -xres {} -yres {} -fx {} -fy {} -cx {} -cy {} -inertia_weight {} -smoothness_weight {}'.format(
    depth2depth_path, input_depth_png, output_depth_path,
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
