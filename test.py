# -*- coding:utf-8 -*-
import SimpleITK
import numpy as np
import os
import model
import utils
import torch
import argparse
from tqdm import tqdm
import data
from model.network import Network
from utils import normal2, unnormal
from measure import compute_measure
from model.layers import PositionalEncoding
from SDVFM.networks import Restormer_Backbonesino
from FDVFM.networks import RestormerBackbonefft

if __name__ == '__main__':

    # -----------------------
    # parameters settings
    # -----------------------
    parser = argparse.ArgumentParser()

    # about ArSSR model
    parser.add_argument('-encoder', type=str, default='ResCNN', dest='encoder_name',
                        help='the type of encoder network, including RDN (default), ResCNN, and SRResnet.')
    parser.add_argument('-encoder_depth', type=int, default=8, dest='encoder_depth',
                        help='the depth of the encoder network (default=8).')
    parser.add_argument('-decoder_depth', type=int, default=10, dest='decoder_depth',
                        help='the depth of the decoder network (default=8).')
    parser.add_argument('-width', type=int, default=320, dest='decoder_width',
                        help='the width of the decoder network (default=256).')
    parser.add_argument('-feature_dim', type=int, default=128, dest='feature_dim',
                        help='the dimension size of the feature vector (default=128)')
    parser.add_argument('-pre_trained_model', type=str, default='./checkpoint/model_param_16.pkl',
                        dest='pre_trained_model', help='the file path of LR input image for testing')

    # about GPU
    parser.add_argument('-is_gpu', type=int, default=1, dest='is_gpu',
                        help='enable GPU (1->enable, 0->disenable)')
    parser.add_argument('-gpu', type=int, default=0, dest='gpu',
                        help='the number of GPU')

    # about file
    parser.add_argument('-input_path', type=str, default=r'G:\deeplearning-code-dataset\Foundation-Models\namiweijing-test/', dest='input_path',
                        help='the file path of LR input image')
    parser.add_argument('-output_path', type=str, default='./output-200-namiweijing', dest='output_path',
                        help='the file save path of reconstructed result')
    parser.add_argument('--norm_max', type=float, default=10., help='the max value of dataset')
    parser.add_argument('--norm_min', type=float, default=0., help='the min value of dataset')

    args = parser.parse_args()
    encoder_name = args.encoder_name
    encoder_depth = args.encoder_depth
    decoder_depth = args.decoder_depth
    decoder_width = args.decoder_width
    feature_dim = args.feature_dim
    pre_trained_model = args.pre_trained_model
    gpu = args.gpu
    is_gpu = args.is_gpu
    input_path = args.input_path
    output_path = args.output_path
    norm_max = args.norm_max
    norm_min = args.norm_min

    x_path = os.path.join(output_path, 'x')
    y_path = os.path.join(output_path, 'y')
    pred_path = os.path.join(output_path, 'pred')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print('Create path : {}'.format(output_path))
    if not os.path.exists(x_path):
        os.makedirs(x_path)
        print('Create path : {}'.format(x_path))
    if not os.path.exists(y_path):
        os.makedirs(y_path)
        print('Create path : {}'.format(y_path))
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
        print('Create path : {}'.format(pred_path))

    # -----------------------
    # model
    # ----------------------')
    if is_gpu == 1 and torch.cuda.is_available():
        DEVICE = torch.device('cuda:{}'.format(str(gpu)))
    else:
        DEVICE = torch.device('cpu')
    network = Network(feature_dim=feature_dim, encoder_depth=encoder_depth, decoder_depth=int(decoder_depth / 2),
                      decoder_width=decoder_width).to(DEVICE)
    network.load_state_dict(torch.load(pre_trained_model, map_location=DEVICE))
    encoding = PositionalEncoding(in_dim=2, frequency_bands=10, include_input=True).to(DEVICE)  # frequency_bands=8

    SVFM = Restormer_Backbonesino().to(DEVICE)
    save_path = 'SDVFM/save-sino/'
    f = os.path.join(save_path, 'model_{}iter.ckpt'.format(100))
    SVFM.load_state_dict(torch.load(f))

    FVFM = RestormerBackbonefft().to(DEVICE)
    save_path = 'FDVFM/save-fft'
    f = os.path.join(save_path, 'model_{}iter.ckpt'.format(100))
    FVFM.load_state_dict(torch.load(f))


    # -----------------------
    # load data
    # -----------------------
    test_loader = data.get_loader(mode='test',  load_mode=0,
                                  saved_path=input_path,
                                  test_patient='400', batch_size=1, num_workers=0)

    # -----------------------
    # SR
    # -----------------------
    pred_ssim_avg, pred_rmse_avg = 0, 0
    pred_ssim_avg1, pred_rmse_avg1 = [], []
    network.eval()
    SVFM.eval()
    FVFM.eval()
    with torch.no_grad():
        for i, (input_img, target_img, xy_input) in enumerate(test_loader):
            input_img_shape = (input_img.shape[1], input_img.shape[2])
            target_img_shape = (target_img.shape[1], target_img.shape[2])
            input_img = normal2(input_img, norm_max=norm_max, norm_min=norm_min)
            target_img = normal2(target_img, norm_max=norm_max, norm_min=norm_min)
            input_img = input_img.unsqueeze(1).to(DEVICE).float()  # N×1×h×w
            target_img = target_img.to(DEVICE).float().view(1, -1).unsqueeze(-1)  # N×K×1
            xy_input = xy_input.view(1, -1, 2).to(DEVICE).float()  # N×K×2
            xy_input_pe = encoding(xy_input).float()
            with torch.no_grad():
                _, feature_sino = SVFM(input_img)
                _, feature_fft = FVFM(torch.fft.fft2(input_img).float())
                feature_ifft = torch.fft.ifft2(feature_fft).float()
            feature = torch.cat((feature_sino, feature_ifft), dim=1)
                # feature = feature_sino
            img_pre = network(xy_input, xy_input_pe, feature.detach())  # N×K×1

            x = unnormal(input_img.view(input_img_shape).cpu().data.clamp(0, 1).detach(), norm_max=norm_max, norm_min=norm_min)
            y = unnormal(target_img.view(target_img_shape).cpu().data.clamp(0, 1).detach(), norm_max=norm_max, norm_min=norm_min)
            pred = unnormal(img_pre.view(target_img_shape).cpu().data.clamp(0, 1).detach(), norm_max=norm_max, norm_min=norm_min)

            # FBP
            # x = fbp.convert(x)
            # y = fbp.convert(y)
            # pred = fbp.convert(pred)

            np.save(os.path.join(output_path, 'x', '{}_result'.format(str(i).zfill(3))), x)
            np.save(os.path.join(output_path, 'y', '{}_result'.format(str(i).zfill(3))), y)
            np.save(os.path.join(output_path, 'pred', '{}_result'.format(str(i).zfill(3))), pred)

            data_range = norm_max - norm_min

            pred_result = compute_measure(y, pred, data_range)
            pred_ssim_avg += pred_result[0]
            pred_ssim_avg1.append(pred_result[0])
            pred_rmse_avg += pred_result[1]
            pred_rmse_avg1.append(pred_result[1])

        print('\n')
        print(
            'After learning\nSSIM avg: {:.8f} \nRMSE avg: {:.8f}'.format(
                pred_ssim_avg / len(test_loader),
                pred_rmse_avg / len(test_loader)))

    # save file

