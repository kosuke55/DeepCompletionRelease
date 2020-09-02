import argparse
import os
from pathlib import Path

import cv2
import numpy as np


input_dir = '../torch/result/bound_realsense_test_bound/'
output_dir = '../torch/result/bound_realsense_weight/'


def _generate_occusion_weight(file, output_dir):
    print(file)
    bound = cv2.imread(file)
    bound = bound / float(bound.max())
    weight = ((1 - bound[:, :, 1]) ** 3 * 1000).astype(np.uint16)
    weight_file = str(
        Path(output_dir) / Path(file).name.replace(
            '_bound_est.png', '_weight.png'))
    cv2.imwrite(weight_file, weight)


def generate_occusion_weight(input_dir, output_dir, idx=None):
    if idx is not None:
        file = str(Path(input_dir) / 'realsense_{:03}_bound_est.png'.format(idx))
        _generate_occusion_weight(file, output_dir)
    else:
        files = list(Path(input_dir).glob('*_bound_est.png'))
        for i, file in enumerate(files):
            _generate_occusion_weight(str(file), output_dir)


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--idx', '-i', type=int,
    help='idx',
    required=True)
args = parser.parse_args()

idx = args.idx

if not os.path.isdir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
generate_occusion_weight(input_dir, output_dir, idx)
