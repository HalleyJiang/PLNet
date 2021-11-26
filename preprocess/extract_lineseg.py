# adapted from https://github.com/svip-lab/Indoor-SfMLearner/blob/master/extract_superpixel.py

import os
import glob

import numpy as np
import cv2
from skimage.color import label2rgb

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import tqdm
from functools import partial

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    help='path to nyu data',
                    required=True)

args = parser.parse_args()

data_path = args.data_path

train_dir = os.path.join(data_path, "nyu2_train")
train_scenes = sorted([name for name in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, name))])

test_dir = os.path.join(data_path, "nyu2_test")
search = test_dir + "/*_colors.png"
test_files = sorted(glob.glob(search))


def extract_lineseg(filename):
    CROP = 16
    image = cv2.imread(filename, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape

    corp_image = image[CROP:-CROP, CROP:-CROP]

    lsd = cv2.createLineSegmentDetector(0, _scale=1)
    lines = lsd.detect(corp_image)[0]
    lines = np.squeeze(lines, 1)
    lengths = np.sqrt((lines[:, 0] - lines[:, 2]) ** 2 + (lines[:, 1] - lines[:, 3]) ** 2)
    arr1inds = lengths.argsort()[::-1]
    lengths = lengths[arr1inds[::-1]]
    lines = lines[arr1inds[::-1]]
    lines = lines[lengths > np.sqrt(corp_image.shape[0]**2+corp_image.shape[1]**2) / 10]
    lines = lines[:min(lines.shape[0], 255)]

    line_seg = np.zeros([h, w], dtype=np.int)

    n = 1
    for k in range(lines.shape[0]):
        x1, y1, x2, y2 = lines[k]

        xmin = max(0, int(np.floor(min(x1, x2))))
        xmax = min(int(np.ceil(max(x1, x2))), w - 2 * CROP)

        ymin = max(0, int(np.floor(min(y1, y2))))
        ymax = min(int(np.floor(max(y1, y2))), h - 2 * CROP)

        points = []
        for i in range(xmin, xmax):
            for j in range(ymin, ymax):
                p = np.array([i, j])
                vec1 = lines[k, :2] - p
                vec2 = lines[k, 2:] - p
                distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(lines[k, :2] - lines[k, 2:])
                if distance < 1:
                    points.append([CROP + j, CROP + i])

        if len(points) < 3:
            continue
        else:
            for p in points:
                line_seg[p[0], p[1]] = n
            n += 1

    return line_seg


def images2seg(scene, filename, index):

    line_seg = extract_lineseg(filename)

    cv2.imwrite(os.path.join(train_dir, scene, index + "_line.png"), line_seg.astype(np.uint8))

    color = label2rgb(line_seg, bg_label=0)
    color = (255 * color).astype(np.uint8)
    cv2.imwrite(os.path.join(train_dir, scene, index + "_line.jpg"), color)

    return


# multi processing fitting
executor = ProcessPoolExecutor(max_workers=cpu_count())
futures = []

for scene in train_scenes:

    search = os.path.join(train_dir, scene) + "/*.jpg"
    files = sorted(glob.glob(os.path.join(os.getcwd(), search)))
    l = len(files)

    for file in files:
        index = file.split('/')[-1].split('.')[0]
        if index.isdigit():
            task = partial(images2seg, scene, file, index)
            futures.append(executor.submit(task))

    results = []
    [results.append(future.result()) for future in tqdm.tqdm(futures)]


for filename in test_files:

    index = int(filename.split('/')[-1].split('_')[0])
    line_seg = extract_lineseg(filename)
    cv2.imwrite(os.path.join(test_dir, "{:05d}_line.png".format(index)), line_seg.astype(np.uint8))

    color = label2rgb(line_seg, bg_label=0)
    color = (255 * color).astype(np.uint8)
    cv2.imwrite(os.path.join(test_dir, "{:05d}_line.jpg".format(index)), color)


