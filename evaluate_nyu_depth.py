from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    log10 = np.mean(np.abs(np.log10(pred / gt)))

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, log10, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-2
    MAX_DEPTH = 10

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        dataset = datasets.NYUDataset(opt.data_path, filenames, encoder_dict['height'], encoder_dict['width'],
                                      [0], 1, is_test=True, return_plane=True, num_plane_keysets=0,
                                      return_line=True, num_line_keysets=0)

        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, [0])

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

        model_dict = depth_decoder.state_dict()
        decoder_dict = torch.load(decoder_path)
        depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in model_dict})
        
        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        gt_depths = []
        planes = []
        lines = []
        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()
                norm_pix_coords = [data[("norm_pix_coords", s)].cuda() for s in opt.scales]

                gt_depth = data["depth_gt"][:, 0].numpy()
                gt_depths.append(gt_depth)

                plane = data[("plane", 0,  -1)][:, 0].numpy()
                planes.append(plane)
                line = data[("line", 0, -1)][:, 0].numpy()
                lines.append(line)

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
                    norm_pix_coords = [torch.cat((pc, torch.flip(pc, [3])), 0) for pc in norm_pix_coords]
                    norm_pix_coords[0][norm_pix_coords[0].shape[0] // 2:, 0] *= -1

                output = depth_decoder(encoder(input_color), norm_pix_coords)

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)

        gt_depths = np.concatenate(gt_depths)
        planes = np.concatenate(planes)
        lines = np.concatenate(lines)
        pred_disps = np.concatenate(pred_disps)

    else:
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        dataset = datasets.NYUDataset(opt.data_path, filenames,  self.opt.height, self.opt.width,
                                      [0], 1, is_test=True, return_plane=True, num_plane_keysets=0,
                                      return_line=True, num_line_keysets=0)

        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        gt_depths = []
        planes = []
        lines = []
        for data in dataloader:
            gt_depth = data["depth_gt"][:, 0].numpy()
            gt_depths.append(gt_depth)
            plane = data[("plane", 0, -1)][:, 0].numpy()
            planes.append(plane)
            line = data[("line", 0, -1)][:, 0].numpy()
            lines.append(line)
        gt_depths = np.concatenate(gt_depths)
        planes = np.concatenate(planes)
        lines = np.concatenate(lines)

        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    print("-> Evaluating")
    print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    gt_plane_pixel_deviations = []
    gt_plane_instance_max_deviations = []
    gt_flatness_ratios = []
    gt_line_pixel_deviations = []
    gt_line_instance_max_deviations = []
    gt_straightness_ratios = []

    pred_plane_pixel_deviations = []
    pred_plane_instance_max_deviations = []
    pred_flatness_ratios = []
    pred_line_pixel_deviations = []
    pred_line_instance_max_deviations = []
    pred_straightness_ratios = []

    norm_pix_coords = dataset.get_norm_pix_coords()

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        crop_mask = np.zeros(mask.shape)
        crop_mask[dataset.default_crop[2]:dataset.default_crop[3], \
        dataset.default_crop[0]:dataset.default_crop[1]] = 1
        mask = np.logical_and(mask, crop_mask)
        mask_pred_depth = pred_depth[mask]
        mask_gt_depth = gt_depth[mask]

        mask_pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(mask_gt_depth) / np.median(mask_pred_depth)
            ratios.append(ratio)
            mask_pred_depth *= ratio
        else:
            ratio = 1
            ratios.append(ratio)

        mask_pred_depth[mask_pred_depth < MIN_DEPTH] = MIN_DEPTH
        mask_pred_depth[mask_pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(mask_gt_depth, mask_pred_depth))

        # compute the flatness and straightness

        plane_seg = planes[i]
        line_seg = lines[i]

        X = norm_pix_coords[0] * gt_depth
        Y = norm_pix_coords[1] * gt_depth
        Z = gt_depth
        for j in range(plane_seg.max()):
            seg_x = X[plane_seg == (j + 1)]
            seg_y = Y[plane_seg == (j + 1)]
            seg_z = Z[plane_seg == (j + 1)]
            P = np.stack((seg_x, seg_y, seg_z), axis=1)

            mean_P = P.mean(axis=0)
            cent_P = P - mean_P
            conv_P = cent_P.T.dot(cent_P) / seg_x.shape[0]

            e_vals, e_vecs = np.linalg.eig(conv_P)
            idx = e_vals.argsort()[::-1]
            e_vals = e_vals[idx]
            e_vecs = e_vecs[:, idx]

            deviations = np.abs(cent_P.dot(e_vecs[:, 2]))
            variance_ratios = e_vals / e_vals.sum()

            gt_plane_instance_max_deviations.append(np.max(deviations))
            gt_plane_pixel_deviations.append(deviations)
            gt_flatness_ratios.append(variance_ratios[2])

        for j in range(line_seg.max()):
            seg_x = X[line_seg == (j + 1)]
            seg_y = Y[line_seg == (j + 1)]
            seg_z = Z[line_seg == (j + 1)]
            P = np.stack((seg_x, seg_y, seg_z), axis=1)
            mean_P = P.mean(axis=0)
            cent_P = P - mean_P

            conv_P = cent_P.T.dot(cent_P) / seg_x.shape[0]

            e_vals, e_vecs = np.linalg.eig(conv_P)
            idx = e_vals.argsort()[::-1]
            e_vals = e_vals[idx]
            e_vecs = e_vecs[:, idx]

            dev2 = np.sum(cent_P ** 2, 1) - (cent_P.dot(e_vecs[:, 0])) ** 2
            dev2[dev2 < 0] = 0
            deviations = np.sqrt(dev2)
            gt_line_instance_max_deviations.append(np.max(deviations))
            gt_line_pixel_deviations.append(deviations)

            variance_ratios = e_vals / e_vals.sum()
            gt_straightness_ratios.append(variance_ratios[1] + variance_ratios[2])

        pred_depth *= ratio
        X = norm_pix_coords[0] * pred_depth
        Y = norm_pix_coords[1] * pred_depth
        Z = pred_depth
        for j in range(plane_seg.max()):
            seg_x = X[plane_seg == (j + 1)]
            seg_y = Y[plane_seg == (j + 1)]
            seg_z = Z[plane_seg == (j + 1)]
            P = np.stack((seg_x, seg_y, seg_z), axis=1)

            mean_P = P.mean(axis=0)
            cent_P = P - mean_P
            conv_P = cent_P.T.dot(cent_P) / seg_x.shape[0]

            e_vals, e_vecs = np.linalg.eig(conv_P)
            idx = e_vals.argsort()[::-1]
            e_vals = e_vals[idx]
            e_vecs = e_vecs[:, idx]

            deviations = np.abs(cent_P.dot(e_vecs[:, 2]))
            variance_ratios = e_vals / e_vals.sum()

            pred_plane_instance_max_deviations.append(np.max(deviations))
            pred_plane_pixel_deviations.append(deviations)
            pred_flatness_ratios.append(variance_ratios[2])

        for j in range(line_seg.max()):
            seg_x = X[line_seg == (j + 1)]
            seg_y = Y[line_seg == (j + 1)]
            seg_z = Z[line_seg == (j + 1)]
            P = np.stack((seg_x, seg_y, seg_z), axis=1)
            mean_P = P.mean(axis=0)
            cent_P = P - mean_P

            conv_P = cent_P.T.dot(cent_P) / seg_x.shape[0]
            e_vals, e_vecs = np.linalg.eig(conv_P)
            idx = e_vals.argsort()[::-1]
            e_vals = e_vals[idx]
            e_vecs = e_vecs[:, idx]

            dev2 = np.sum(cent_P ** 2, 1) - (cent_P.dot(e_vecs[:, 0])) ** 2
            dev2[dev2 < 0] = 0
            deviations = np.sqrt(dev2)
            pred_line_instance_max_deviations.append(np.max(deviations))
            pred_line_pixel_deviations.append(deviations)

            variance_ratios = e_vals / e_vals.sum()
            pred_straightness_ratios.append(variance_ratios[1] + variance_ratios[2])

    mean_errors = np.array(errors).mean(0)
    result_path = os.path.join(opt.load_weights_folder, "result_{}_split.txt".format(opt.eval_split))
    f = open(result_path, 'w+')

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)), file=f)

    print("\n  " + ("{:>8} | " * 8).format("abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\")

    print("\n  " + ("{:>8} | " * 8).format("abs_rel", "sq_rel", "rmse", "rmse_log", "log10",  "a1", "a2", "a3"), file=f)
    print(("&{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\", file=f)

    for th in [0.3]:
        print(th)
        print(th, file=f)

        mask_gt_plane_pixel_deviations = \
            [i for i, j in zip(gt_plane_pixel_deviations, gt_plane_instance_max_deviations) if j < th]
        mask_gt_plane_instance_max_deviations = [j for j in gt_plane_instance_max_deviations if j < th]
        mask_gt_flatness_ratios = \
            [i for i, j in zip(gt_flatness_ratios, gt_plane_instance_max_deviations) if j < th]

        mask_gt_plane_pixel_deviations = np.concatenate(mask_gt_plane_pixel_deviations)
        mask_gt_plane_instance_max_deviations = np.array(mask_gt_plane_instance_max_deviations)
        mask_gt_flatness_ratios = np.array(mask_gt_flatness_ratios)
        gt_plane_mean_dev = mask_gt_plane_pixel_deviations.mean()
        gt_plane_max_dev = mask_gt_plane_instance_max_deviations.mean()
        gt_flatness_ratio = mask_gt_flatness_ratios.mean()

        mask_gt_line_pixel_deviations = \
            [i for i, j in zip(gt_line_pixel_deviations, gt_line_instance_max_deviations) if j < th]
        mask_gt_line_instance_max_deviations = [j for j in gt_line_instance_max_deviations if j < th]
        mask_gt_straightness_ratios = \
            [i for i, j in zip(gt_straightness_ratios, gt_line_instance_max_deviations) if j < th]

        mask_gt_line_pixel_deviations = np.concatenate(mask_gt_line_pixel_deviations)
        mask_gt_line_instance_max_deviations = np.array(mask_gt_line_instance_max_deviations)
        mask_gt_straightness_ratios = np.array(mask_gt_straightness_ratios)

        gt_line_mean_dev = mask_gt_line_pixel_deviations.mean()
        gt_line_max_dev = mask_gt_line_instance_max_deviations.mean()
        gt_straightness_ratio = mask_gt_straightness_ratios.mean()

        print("\n  GT :   " + ("{:>12} | " * 6).format(
            "plane mean dev", "plane max dev", "flatness ratio", "line mean dev", "line max dev", "straightness ratio"))
        print(" "*10 + ("&{: 10.6f}    " * 6).format(gt_plane_mean_dev, gt_plane_max_dev, gt_flatness_ratio,
                                                   gt_line_mean_dev, gt_line_max_dev, gt_straightness_ratio) + "\\\\")

        print("\n  GT :   " + ("{:>12} | " * 6).format("plane mean dev",
                    "plane max dev", "flatness ratio", "line mean dev", "line max dev", "straightness ratio"), file=f)
        print(" " * 10 + ("&{: 10.6f}    " * 6).format(gt_plane_mean_dev, gt_plane_max_dev, gt_flatness_ratio,
                                            gt_line_mean_dev, gt_line_max_dev, gt_straightness_ratio) + "\\\\", file=f)

        mask_pred_plane_pixel_deviations = \
            [i for i, j in zip(pred_plane_pixel_deviations, gt_plane_instance_max_deviations) if j < th]
        mask_pred_plane_instance_max_deviations = \
            [i for i, j in zip(pred_plane_instance_max_deviations, gt_plane_instance_max_deviations) if j < th]
        mask_pred_flatness_ratios = \
            [i for i, j in zip(pred_flatness_ratios, gt_plane_instance_max_deviations) if j < th]

        mask_pred_plane_pixel_deviations = np.concatenate(mask_pred_plane_pixel_deviations)
        mask_pred_plane_instance_max_deviations = np.array(mask_pred_plane_instance_max_deviations)
        mask_pred_flatness_ratios = np.array(mask_pred_flatness_ratios)
        pred_plane_mean_dev = mask_pred_plane_pixel_deviations.mean()
        pred_plane_max_dev = mask_pred_plane_instance_max_deviations.mean()
        pred_flatness_ratio = mask_pred_flatness_ratios.mean()

        mask_pred_line_pixel_deviations = \
            [i for i, j in zip(pred_line_pixel_deviations, gt_line_instance_max_deviations) if j < th]
        mask_pred_line_instance_max_deviations = \
            [i for i, j in zip(pred_line_instance_max_deviations, gt_line_instance_max_deviations) if j < th]
        mask_pred_straightness_ratios = \
            [i for i, j in zip(pred_straightness_ratios, gt_line_instance_max_deviations) if j < th]

        mask_pred_line_pixel_deviations = np.concatenate(mask_pred_line_pixel_deviations)
        mask_pred_line_instance_max_deviations = np.array(mask_pred_line_instance_max_deviations)
        mask_pred_straightness_ratios = np.array(mask_pred_straightness_ratios)

        pred_line_mean_dev = mask_pred_line_pixel_deviations.mean()
        pred_line_max_dev = mask_pred_line_instance_max_deviations.mean()
        pred_straightness_ratio = mask_pred_straightness_ratios.mean()

        print("\n  Pred: " + ("{:>12} | " * 6).format(
            "plane mean dev", "plane max dev", "flatness ratio", "line mean dev", "line max dev", "straightness ratio"))
        print(" " * 12 + ("&{: 10.6f}    " * 6).format(pred_plane_mean_dev, pred_plane_max_dev, pred_flatness_ratio,
                                               pred_line_mean_dev, pred_line_max_dev, pred_straightness_ratio) + "\\\\")

        print("\n  Pred: " + ("{:>12} | " * 6).format("plane mean dev",
                                                                             "plane max dev", "flatness ratio",
                                                                             "line mean dev", "line max dev",
                                                                             "straightness ratio"), file=f)
        print(" " * 12 + ("&{: 10.6f}    " * 6).format(pred_plane_mean_dev, pred_plane_max_dev, pred_flatness_ratio,
                                                     pred_line_mean_dev, pred_line_max_dev,
                                                     pred_straightness_ratio) + "\\\\", file=f)
    print("\n-> Done!")
    print("\n-> Done!", file=f)


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
