import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

import lpips
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid


'''
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''

torch.backends.cudnn.enabled = False


def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))



def main(json_path='options/train_msrresnet_psnr.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist


    writer = SummaryWriter('./runs/' + opt['task'])

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-

    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)
    
    # current_step = 0

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                # train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'])
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())


    # ==================================================================
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    best_PSNRY = 0
    best_step = 0    
    # ==================================================================


    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    for epoch in range(1000000):  # keep running
        if opt['dist']:
            train_sampler.set_epoch(epoch)

        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                    # ----------------------------------------
                    writer.add_scalar('loss', v, global_step=current_step)
                    # ----------------------------------------
                    
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

                avg_psnr = 0.0
                avg_ssim = 0.0
                avg_psnrY = 0.0
                avg_ssimY = 0.0
                avg_lpips = 0.0                
                idx = 0
                save_list = []

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

                    # -----------------------
                    # save estimated image E
                    # -----------------------
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                    util.imsave(E_img, save_img_path)

                    # -----------------------
                    # calculate PSNR
                    # -----------------------
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)

                    # ==================================================================
                    current_ssim = util.calculate_ssim(E_img, H_img, border=border)
                    current_lpips = loss_fn_alex(im2tensor(E_img).cuda(), im2tensor(H_img).cuda()).item()
                    output_y = util.bgr2ycbcr(E_img.astype(np.float32) / 255.) * 255.
                    img_gt_y = util.bgr2ycbcr(H_img.astype(np.float32) / 255.) * 255.
                    psnr_y = util.calculate_psnr(output_y, img_gt_y, border=border)
                    ssim_y = util.calculate_ssim(output_y, img_gt_y, border=border)

                    # ==================================================================
                    logger.info('{:->4d}--> {:>20s} | PSNR: {:<4.2f}, SSIM: {:<5.4f}, PSNRY: {:<4.2f}, SSIMY: {:<5.4f}, LPIPS: {:<5.4f},'.format(idx, image_name_ext, current_psnr, current_ssim, psnr_y, ssim_y, current_lpips))

                    # logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))
                    
                    avg_psnr += current_psnr
                    avg_ssim += current_ssim
                    avg_psnrY += psnr_y
                    avg_ssimY += ssim_y
                    avg_lpips += current_lpips

                    if img_name in opt['train']['save_image']:
                        print(img_name)
                        save_list.append(util.uint2tensor3(E_img)[:, :512, :512])


                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                avg_psnrY = avg_psnrY / idx
                avg_ssimY = avg_ssimY / idx
                avg_lpips = avg_lpips / idx

                if len(save_list) > 0 and current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                    save_images = make_grid(save_list, nrow=len(save_list))
                    writer.add_image("test", save_images, global_step=current_step)




                #     avg_psnr += current_psnr

                # avg_psnr = avg_psnr / idx

                if avg_psnrY >= best_PSNRY:
                    best_step = current_step
                    best_PSNRY = avg_psnrY
                    
                # testing log
                # logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average: PSNR: {:<.2f}, SSIM: {:<.4f}, PSNRY: {:<.2f}, SSIMY: {:<.4f}, LPIPS: {:<.4f}'.format(epoch, current_step, avg_psnr, avg_ssim, avg_psnrY, avg_ssimY, avg_lpips))
                logger.info('--- best PSNRY --->   iter:{:8,d}, Average: PSNR: {:<.2f}\n'.format(best_step, best_PSNRY))

                writer.add_scalar('PSNRY', avg_psnrY, global_step=current_step)
                writer.add_scalar('SSIMY', avg_ssimY, global_step=current_step)
                writer.add_scalar('PSNR',  avg_psnr,  global_step=current_step)
                writer.add_scalar('SSIM',  avg_ssim,  global_step=current_step)
                writer.add_scalar('LPIPS', avg_lpips, global_step=current_step)


if __name__ == '__main__':
    main()
