# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: train.py
#   Author  : Qing Wu
#   Email   : wuqing@shanghaitech.edu.cn
#   Date    : 2021/9/19
# -----------------------------------------
import os
import data
import torch
import model
import argparse
import time
from torch.utils.tensorboard import SummaryWriter
from model.network import Network
from skimage.transform import iradon
import numpy as np
from torch.optim import lr_scheduler
from utils import normal2
import torch
import random
# import tensorflow as tf
import numpy as np
from model.layers import PositionalEncoding
from DegAEsino.networks import Restormer_Backbonesino
from DegAEfft.networks import RestormerBackbonefft

# # 设置随机种子
# seed = 42
#
#
# np.random.seed(seed)# 设置 NumPy 中的随机种子
# random.seed(seed) #设置 Python 标准库中的随机种子，以确保其他 Python 函数中使用的随机数也是可复现的。
#
# tf.random.set_seed(seed) #设置 TensorFlow 中的随机种子
#
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed) #设置 PyTorch 在 CUDA 环境下的随机种子，以确保 CUDA 计算的结果是可复现的。
# torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU，此命令将确保所有的 GPU 使用相同的随机种子。
# torch.backends.cudnn.deterministic = True # 确保在使用 cuDNN 加速时结果可复现，但可能会降低性能。
# torch.backends.cudnn.benchmark = False #禁用 cuDNN 的自动寻找最适合当前配置的高效算法的功能，以确保结果的一致性。


if __name__ == '__main__':

    writer = SummaryWriter('./log-20layer')

    # -----------------------
    # parameters settings
    # -----------------------
    parser = argparse.ArgumentParser()

    # about ArSSR model
    parser.add_argument('-encoder_name', type=str, default='ResCNN', dest='encoder_name',
                        help='the type of encoder network, including RDN (default), ResCNN, and SRResnet.')
    parser.add_argument('-encoder_depth', type=int, default=8, dest='encoder_depth',
                        help='the depth of the encoder network (default=1).')
    parser.add_argument('-decoder_depth', type=int, default=10, dest='decoder_depth',
                        help='the depth of the decoder network (default=8).')
    parser.add_argument('-decoder_width', type=int, default=320, dest='decoder_width',
                        help='the width of the decoder network (default=256).')
    parser.add_argument('-feature_dim', type=int, default=128, dest='feature_dim',
                        help='the dimension size of the feature vector (default=128)')

    # about training and validation data
    parser.add_argument('--mode', type=str, default='train', help="train | test")
    parser.add_argument('--load_mode', type=int, default=0)
    parser.add_argument('-hr_data_train', type=str, default='/data/wangjiping/ge-reconstruction/data/train2/mix', dest='hr_data_train',
                        help='the file path of HR patches for training')
    parser.add_argument('-hr_data_val', type=str, default='/data/wangjiping/ge-reconstruction/data/test/286', dest='hr_data_val',
                        help='the file path of HR patches for validation')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--norm_max', type=float, default=10., help='the max value of dataset')
    parser.add_argument('--norm_min', type=float, default=0., help='the min value of dataset')

    # about training hyper-parameters
    parser.add_argument('-lr', type=float, default=1e-5, dest='lr',
                        help='the initial learning rate')
    parser.add_argument('-lr_decay_epoch', type=int, default=20, dest='lr_decay_epoch',
                        help='learning rate multiply by 0.5 per lr_decay_epoch.')
    parser.add_argument('-epoch', type=int, default=100, dest='epoch',
                        help='the total number of epochs for training')
    parser.add_argument('-summary_epoch', type=int, default=1, dest='summary_epoch',
                        help='the current model will be saved per summary_epoch')
    parser.add_argument('-bs', type=int, default=1, dest='batch_size',
                        help='the number of LR-HR patch pairs (i.e., N in Equ. 3)')
    parser.add_argument('-gpu', type=int, default=2, dest='gpu',
                        help='the number of GPU')
    parser.add_argument('-checkpoint_path', type=str, default='./checkpoint-20layer', dest='checkpoint_path',
                        help='the file path of checkpoint')

    args = parser.parse_args()
    encoder_name = args.encoder_name
    encoder_depth = args.encoder_depth
    decoder_depth = args.decoder_depth
    decoder_width = args.decoder_width
    feature_dim = args.feature_dim
    mode = args.mode
    load_mode = args.load_mode
    hr_data_train = args.hr_data_train
    hr_data_val = args.hr_data_val
    num_workers = args.num_workers
    norm_max = args.norm_max
    norm_min = args.norm_min
    lr = args.lr
    lr_decay_epoch = args.lr_decay_epoch
    epoch = args.epoch
    summary_epoch = args.summary_epoch
    batch_size = args.batch_size
    gpu = args.gpu
    checkpoint_path = args.checkpoint_path

    # -----------------------
    # display parameters
    # -----------------------
    print('Parameter Settings')
    print('')
    print('------------File------------')
    print('mode: {}'.format(mode))
    print('load_mode: {}'.format(load_mode))
    print('data_train: {}'.format(hr_data_train))
    print('data_val: {}'.format(hr_data_val))
    print('norm_max: {}'.format(norm_max))
    print('norm_min: {}'.format(norm_min))
    print('------------Train-----------')
    print('lr: {}'.format(lr))
    print('batch_size_train: {}'.format(batch_size))
    print('gpu: {}'.format(gpu))
    print('epochs: {}'.format(epoch))
    print('summary_epoch: {}'.format(summary_epoch))
    print('lr_decay_epoch: {}'.format(lr_decay_epoch))
    print('checkpoint_path: {}'.format(checkpoint_path))
    print('------------Model-----------')
    print('encoder_name : {}'.format(encoder_name))
    print('decoder feature_dim: {}'.format(feature_dim))
    print('encoder depth: {}'.format(encoder_depth))
    print('decoder depth: {}'.format(decoder_depth))
    print('decoder width: {}'.format(decoder_width))
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        print('Create path : {}'.format(checkpoint_path))
    for i in range(5):
        print(i + 1, end="s,")
        time.sleep(1)

    # -----------------------
    # load data
    # -----------------------
    train_loader = data.get_loader(mode=mode,  load_mode=load_mode, saved_path=hr_data_train,
                                   test_patient='TEST', batch_size=batch_size, num_workers=num_workers)
    val_loader = data.get_loader(mode='test', load_mode=load_mode, saved_path=hr_data_val,
                                 test_patient='286', batch_size=1, num_workers=num_workers)

    print('\n')
    print('len(train_loader)=', len(train_loader))
    print('len(val_loader)=', len(val_loader))

    # -----------------------
    # model & optimizer
    # -----------------------
    DEVICE = torch.device('cuda:{}'.format(str(gpu) if torch.cuda.is_available() else 'cpu'))
    network = Network(feature_dim=feature_dim, encoder_depth=encoder_depth, decoder_depth=int(decoder_depth / 2),
                      decoder_width=decoder_width).to(DEVICE)

    SVFM = Restormer_Backbonesino().to(DEVICE)
    save_path = './DegAEsino/save-sino/'
    f = os.path.join(save_path, 'model_{}iter.ckpt'.format(100))
    SVFM.load_state_dict(torch.load(f))

    FVFM = RestormerBackbonefft().to(DEVICE)
    save_path = './DegAEfft/save-fft'
    f = os.path.join(save_path, 'model_{}iter.ckpt'.format(100))
    FVFM.load_state_dict(torch.load(f))

    encoding = PositionalEncoding(in_dim=2, frequency_bands=10, include_input=True).to(DEVICE)  # frequency_bands=8
    loss_fun = torch.nn.L1Loss()
    # loss_fun = torch.nn.KLDivLoss()
    optimizer = torch.optim.AdamW(params=network.parameters(), lr=lr)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_epoch, gamma=0.5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epoch, eta_min=1e-6)

    # -----------------------
    # training & validation
    # -----------------------
    for e in range(0, epoch):
        network.train()
        SVFM.eval()
        FVFM.eval()
        loss_train = 0
        for i, (input_img, _, xy_input) in enumerate(train_loader):
            # forward
            # outputsize = input_img.shape[-1]
            input_img = normal2(input_img, norm_max=norm_max, norm_min=norm_min)
            sinogram1 = input_img.unsqueeze(1).to(DEVICE).float()  # N×1×h×w
            sinogram2 = input_img.to(DEVICE).float().view(batch_size, -1).unsqueeze(-1)  # N×K×1
            xy_input = xy_input.view(batch_size, -1, 2).to(DEVICE).float()  # N×K×2
            xy_input_pe = encoding(xy_input).float()
            # print(sinogram1.shape)
            with torch.no_grad():
                _, feature_sino = SVFM(sinogram1)
                _, feature_fft = FVFM(torch.fft.fft2(sinogram1).float())
                feature_ifft = torch.fft.ifft2(feature_fft).float()
                # print(feature.shape)
            feature = torch.cat((feature_sino, feature_ifft), dim=1)
            # feature = feature_sino
            # print(xy_input_pe.size())
            img_pre = network(xy_input, xy_input_pe, feature.detach())  # N×K×1
            # print(img_pre.size())
            img_pre_fft = torch.fft.fft(img_pre, dim=1)
            target_img_fft = torch.fft.fft(sinogram2, dim=1)
            loss = loss_fun(img_pre, sinogram2) + 1e-3 * loss_fun(img_pre_fft, target_img_fft)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # record and print loss
            loss_train += loss.item()
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            print('(TRAIN) Epoch[{}/{}], Steps[{}/{}], Lr:{}, Loss:{:.10f}'.format(e + 1,
                                                                                   epoch,
                                                                                   i + 1,
                                                                                   len(train_loader),
                                                                                   current_lr,
                                                                                   loss.item()))
        # learning rate decays by half every some epochs.
        scheduler.step()
        print('(TRAIN) Epoch[{}/{}], train_Loss:{:.10f}'.format(e + 1,
                                                                epoch,
                                                                (loss_train / len(train_loader))))
        writer.add_scalar('MES_train', loss_train / len(train_loader), e + 1)
        # release memory
        sinogram1 = None
        sinogram2 = None
        xy_input = None
        img_pre = None

        network.eval()
        SVFM.eval()
        FVFM.eval()
        with torch.no_grad():
            loss_val = 0
            for i, (input_img, target_img, xy_input) in enumerate(val_loader):
                input_img = normal2(input_img, norm_max=norm_max, norm_min=norm_min)
                target_img = normal2(target_img, norm_max=norm_max, norm_min=norm_min)
                input_img = input_img.unsqueeze(1).to(DEVICE).float()  # N×1×h×w
                target_img = target_img.to(DEVICE).float().view(1, -1).unsqueeze(-1)  # N×K×1
                xy_input = xy_input.view(1, -1, 2).to(DEVICE).float()  # N×K×2
                xy_input_pe = encoding(xy_input).float()
                _, feature_sino = SVFM(input_img)
                _, feature_fft = FVFM(torch.fft.fft2(input_img).float())
                feature_ifft = torch.fft.ifft2(feature_fft).float()
                feature = torch.cat((feature_sino, feature_ifft), dim=1)
                # feature = feature_sino
                img_pre = network(xy_input, xy_input_pe, feature.detach())  # N×K×1
                # print('sinogram.shape=', sinogram.size())
                # print('sinogram_full.shape=', sinogram_full.size())
                # print('xy_sinogram.shape=', xy_sinogram.size())
                # print('img_pre.shape=', img_pre.size())
                loss_val += loss_fun(img_pre, target_img)
                # save validation
                if (e + 1) % summary_epoch == 0:
                    # save model
                    torch.save(network.state_dict(), os.path.join(checkpoint_path, 'model_param_{}.pkl'.format(e + 1)))

                # writer.add_image('val', img_pre, 1)

        print('(TRAIN) Epoch[{}/{}], val_Loss:{:.10f}'.format(e + 1,
                                                              epoch,
                                                              (loss_val / len(val_loader))))

        writer.add_scalar('MES_val', loss_val / len(val_loader), e + 1)
        # release memory
        input_imgc = None
        target_imgc = None
        xy_input = None
        img_pre = None

    writer.flush()
