# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 is_test=False,
                 return_plane=False,
                 num_plane_keysets=512,
                 return_line=False,
                 num_line_keysets=128,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.is_test = is_test
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()
        self.load_plane = return_plane and self.check_plane()
        self.load_line = return_line and self.check_line()

        if self.load_plane or self.load_line:
            self.pl_resize = {}
            self.num_plane_keysets = num_plane_keysets
            self.num_line_keysets = num_line_keysets
            for i in range(self.num_scales):
                s = 2 ** i
                self.pl_resize[i] = transforms.Resize((self.height // s, self.width // s),
                                                      interpolation=Image.NEAREST)

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

            if "plane" in k or "line" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.pl_resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                if i == -1:
                    continue
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

            if "plane" in k or "line" in k:
                n, im, i = k
                if i == -1 and not self.is_test:
                    continue

                f = np.expand_dims(np.array(f), 0)
                inputs[(n, im, i)] = torch.from_numpy(f).long()

                num_struct_keysets = self.num_plane_keysets if n == "plane" else self.num_line_keysets
                if num_struct_keysets == 0:
                    continue

                keyset_samples = 4 if n == "plane" else 3

                f = f.flatten()
                num_struct_pixels = np.sum(f > 0)
                num_struct = f.max()

                keysets = []
                acc_num = 0
                for j in range(num_struct):
                    pixels_j = np.sum(f == j+1)
                    num_j = int(np.ceil(num_struct_keysets//2**i*pixels_j/num_struct_pixels))
                    acc_num += num_j

                    inx_j = np.argwhere(f == j+1)
                    keyset_j = inx_j[np.random.randint(inx_j.shape[0], size=num_j*keyset_samples)]
                    keyset_j = np.reshape(keyset_j, (keyset_samples, num_j))
                    if keyset_j.size > 0:
                      keysets.append(keyset_j)
                
                if len(keysets):
                    keysets = np.concatenate(keysets, axis=1)
                    keysets = keysets[:, np.random.randint(keysets.shape[1], size=num_struct_keysets//2**i)]
                else:
                    # no a keyset found (no detected planes or lines)
                    keysets = np.zeros((keyset_samples, num_struct_keysets//2**i), dtype=np.int)

                inputs[(n + "_keysets", im, i)] = torch.from_numpy(keysets).long()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) >= 2:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()
            if do_flip:
                K[0, 2] = 1 - K[0, 2]

            width = self.width // (2 ** scale)
            height = self.height // (2 ** scale)

            K[0, :] *= width
            K[1, :] *= height

            inputs[("K", scale)] = torch.from_numpy(K)

            Us, Vs = np.meshgrid(np.linspace(0, width-1, width, dtype=np.float32),
                                 np.linspace(0, height-1, height, dtype=np.float32),
                                 indexing='xy')
            Ones = np.ones([height, width], dtype=np.float32)
            norm_pix_coords = np.stack(((Us - K[0, 2]) / K[0, 0], (Vs - K[1, 2]) / K[1, 1], Ones), axis=0)
            inputs[("norm_pix_coords", scale)] = torch.from_numpy(norm_pix_coords)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if self.load_plane:
            inputs[("plane", 0, -1)] = self.get_plane(folder, frame_index, side, do_flip)

        if self.load_line:
            inputs[("line", 0, -1)] = self.get_line(folder, frame_index, side, do_flip)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            if self.is_test:
                inputs[("color", i, -1)] = self.to_tensor(inputs[("color", i, -1)])
            else:
                del inputs[("color", i, -1)]

        if self.load_plane and not self.is_test:
            del inputs[("plane", 0, -1)]

        if self.load_line and not self.is_test:
            del inputs[("line", 0, -1)]

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_plane(self):
        raise NotImplementedError

    def get_plane(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_line(self):
        raise NotImplementedError

    def get_line(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
