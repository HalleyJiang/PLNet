# adapted from https://github.com/svip-lab/Indoor-SfMLearner/blob/master/extract_superpixel.py

import os
import glob

import numpy as np
import cv2
from skimage.segmentation import slic, watershed, felzenszwalb
from skimage.util import img_as_float
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


def extract_superpixel(filename):
    CROP = 16
    image = cv2.imread(filename)
    h, w, c = image.shape

    corp_image = image[CROP:-CROP, CROP:-CROP, :]

    resize_image = cv2.resize(corp_image, (384, 288))
    resize_image = img_as_float(resize_image)

    segment = felzenszwalb(resize_image, scale=100, sigma=0.5, min_size=50)

    n = 1
    for i in range(segment.max()):
        npixel = np.sum(segment == (i + 1))
        if npixel > 1000:
            segment[segment == (i + 1)] = n
            n += 1
        else:
            segment[segment == (i + 1)] = 0

    segment = cv2.resize(segment, (w - 2 * CROP, h - 2 * CROP), interpolation=cv2.INTER_NEAREST)

    ext_seg = np.zeros([h, w], dtype=np.int)
    ext_seg[CROP:-CROP, CROP:-CROP] = segment

    return ext_seg


def images2seg(scene, filename, index):

    segment = extract_superpixel(filename)
    cv2.imwrite(os.path.join(train_dir, scene, index + "_seg.png"), segment.astype(np.uint8))

    color = label2rgb(segment, bg_label=0)
    color = (255 * color).astype(np.uint8)
    cv2.imwrite(os.path.join(train_dir, scene, index + "_seg.jpg"), color)

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
    segment = extract_superpixel(filename)
    cv2.imwrite(os.path.join(test_dir, "{:05d}_seg.png".format(index)), segment.astype(np.uint8))

    color = label2rgb(segment, bg_label=0)
    color = (255 * color).astype(np.uint8)
    cv2.imwrite(os.path.join(test_dir, "{:05d}_seg.jpg".format(index)), color)
