from skimage.exposure import match_histograms
import cv2
import numpy as np
import bm3d
import glob
import lpips
from utils import utils_image as util
from collections import OrderedDict
import os
import torch
from rich.progress import track



loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores

def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


def eval(path, ref='/home/haoyuc/Research/Data/TestSet/McM/HR'):
    img_paths = glob.glob(os.path.join(path, '*.jpg')) + \
                glob.glob(os.path.join(path, '*.png')) + \
                glob.glob(os.path.join(path, '*.tif'))
    ref_paths = glob.glob(os.path.join(ref, '*.jpg')) + \
                glob.glob(os.path.join(ref, '*.png')) + \
                glob.glob(os.path.join(ref, '*.tif'))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnr_b'] = []
    test_results['lpips'] = []

    img_paths = sorted(img_paths)
    ref_paths = sorted(ref_paths)
    psnr, ssim, psnr_y, ssim_y, psnr_b, lpips_ = 0, 0, 0, 0, 0, 0


    for i in track(range(18)):
        # print(img_paths[i])
        # print(ref_paths[i])
        output = cv2.imread(img_paths[i])
        img_gt = cv2.imread(ref_paths[i])
        # img2 = cv2.imread(path)
        # matched = match_histograms(img2, img1).astype(np.uint8)
        # save = f"/home/haoyuc/Research/BSR/KAIR/results/bm3d_mix2/img_{str(i+1).zfill(3)}_x1.png"
        # cv2.imwrite(save, matched)

        # output = bm3d.bm3d(output, sigma_psd=15/255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
        # output = bm3d.bm3d(output)
        # print(output.shape)
        # print(img_gt.shape)
        # cv2.imwrite(save, output)

        border = 0
        psnr = util.calculate_psnr(output, img_gt, border=border)
        ssim = util.calculate_ssim(output, img_gt, border=border)
        lpips_ = loss_fn_alex(im2tensor(output).cuda(), im2tensor(img_gt).cuda()).item()
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        test_results['lpips'].append(lpips_)    

        # output_y = util.bgr2ycbcr(output.astype(np.float32) / 255.) * 255.
        # img_gt_y = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
        # psnr_y = util.calculate_psnr(output_y, img_gt_y, border=border)
        # ssim_y = util.calculate_ssim(output_y, img_gt_y, border=border)
        # test_results['psnr_y'].append(psnr_y)
        # test_results['ssim_y'].append(ssim_y)

        # idx = i
        # imgname = '1'
        # print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}; '
        #         'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}; '
        #         'PSNR_B: {:.2f} dB; LPIPS: {:.4f}'.
        #         format(idx, imgname, psnr, ssim, psnr_y, ssim_y, psnr_b, lpips_))


    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    ave_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
    # ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
    # ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
    print('-- Average PSNR/SSIM/LPIPS: {:.2f}/{:.4f}/{:.4f}'.format(ave_psnr, ave_ssim, ave_lpips))
    # print('-- Average PSNR_Y/SSIM_Y/LPIPS: {:.2f}/{:.4f}/{:.4f}'.format(ave_psnr_y, ave_ssim_y, ave_lpips))




# path1='E:/Vscode/NewData/Paper3Data/visible-infrared/3/visible1.tif'  
# path2='E:/Vscode/NewData/Paper3Data/visible-infrared/3/infrared1.tif'
# img1=cv2.imread(path1)
# img2=cv2.imread(path2)
# matched =match_histograms(img2,img1)
# cv2.namedWindow('1',cv2.WINDOW_NORMAL)
# cv2.imshow('1',img1)
# cv2.namedWindow('2',cv2.WINDOW_NORMAL)
# cv2.imshow('2',img2)
# cv2.namedWindow('match_img12',cv2.WINDOW_NORMAL)
# cv2.imshow('match_img12',matched.astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()


paths = os.listdir('/home/haoyuc/Research/Code/Restormer-main/results/non_blind')
paths = sorted(paths)
for i in paths:
    if i.startswith('McM') or i.startswith('McM'):
    # if 1:
        path = f"/home/haoyuc/Research/Code/Restormer-main/results/non_blind/{i}/15/"
        print(path)
        eval(path)
        print()
