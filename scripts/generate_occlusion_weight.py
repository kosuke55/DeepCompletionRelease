import os
from pathlib import Path

import cv2
import numpy as np


def generate_occusion_weight(input_dir, output_dir):
    files = list(Path(input_dir).glob('*_bound_est.png'))

    for i, file in enumerate(files):
        bound = cv2.imread(str(file))
        weight = ((1 - bound[:, :, 2]) ** 3 * 1000).astype(np.uint16)
        weight_file = str(
            Path(output_dir) / file.name.replace(
                '_bound_est.png', '_weight.png'))
        cv2.imwrite(weight_file, weight)


if __name__ == "__main__":
    input_dir = '../torch/result/bound_realsense_test_bound/'
    output_dir = '../torch/result/bound_realsense_weight/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    generate_occusion_weight(input_dir, output_dir)
