import os
import torch
import torchvision
import time
import tqdm
from torchsummary import summary
from model.network  import Network
import utils
from model.layers import PositionalEncoding
from thop import profile, clever_format
from DegAEsino.networks import Restormer_Backbonesino
from DegAEfft.networks import RestormerBackbonefft


def calcGPUTime():
    device = 'cuda:0'
    model = Network(feature_dim=128, encoder_depth=8, decoder_depth=int(10 / 2),
                      decoder_width=320)
    model.to(device)
    model.eval()

    # input_img_shape = (150, 912)
    # target_img_shape = (1000, 912)
    input_img = torch.randn(1, 150, 912).to(device)
    # target_img = torch.randn(1, 1000, 912).to(device).view(1, -1).unsqueeze(-1)
    xy_input = utils.make_coord([100, 912], flatten=True)
    SVFM = Restormer_Backbonesino().to(device)
    save_path = './DegAEsino/save-sino/'
    f = os.path.join(save_path, 'model_{}iter.ckpt'.format(100))
    SVFM.load_state_dict(torch.load(f))

    FVFM = RestormerBackbonefft().to(device)
    save_path = './DegAEfft/save-fft'
    f = os.path.join(save_path, 'model_{}iter.ckpt'.format(100))
    FVFM.load_state_dict(torch.load(f))
    with torch.no_grad():
        _, feature_sino = SVFM(torch.randn(1, 1, 150, 912).to(device))
        _, feature_fft = FVFM(torch.fft.fft2(torch.randn(1, 1, 150, 912).to(device)).float())
        feature_ifft = torch.fft.ifft2(feature_fft).float()
    feature = torch.cat((feature_sino, feature_ifft), dim=1)

    # feature = torch.randn(1, 192, 150, 912).to(device)
    # input_img = input_img.unsqueeze(1).to(DEVICE).float()  # N×1×h×w
    # target_img = target_img.to(DEVICE).float().view(1, -1).unsqueeze(-1)  # N×K×1
    xy_input = xy_input.view(1, -1, 2).to(device).float()  # N×K×2
    encoding = PositionalEncoding(in_dim=2, frequency_bands=10, include_input=True).to(device)  # frequency_bands=8
    xy_input_pe = encoding(xy_input).float()

    flops, params = profile(model, inputs=(xy_input, xy_input_pe, feature.detach(), ))
    flops, params = clever_format([flops, params], '%.4f')
    print(f'FLOPs: {flops}')
    print(f'Parameters: {params}')
    # summary(model, input_size=(3, 224, 224), device="cuda")
    # dummy_input1 = torch.randn(1, 1, 512, 512).to(device)
    # dummy_input2 = torch.randn(1, 1, 512, 512).to(device)

    num_iterations = 1000  # 迭代次数
    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(100):
            _ = model(xy_input, xy_input_pe, feature.detach())

    print('testing ...\n')
    total_forward_time = 0.0  # 使用time来测试
    # 记录开始时间
    start_event = time.time() * 1000
    with torch.no_grad():
        for _ in tqdm.tqdm(range(num_iterations)):
            start_forward_time = time.time()
            _ = model(xy_input, xy_input_pe, feature.detach())
            end_forward_time = time.time()
            forward_time = end_forward_time - start_forward_time
            total_forward_time += forward_time * 1000  # 转换为毫秒

    # 记录结束时间
    end_event = time.time() * 1000

    elapsed_time = (end_event - start_event) / 1000.0  # 转换为秒
    fps = num_iterations / elapsed_time

    elapsed_time_ms = elapsed_time / (num_iterations * input_img.shape[0])

    avg_forward_time = total_forward_time / (num_iterations * input_img.shape[0])

    print(f"FPS: {fps}")
    print("elapsed_time_ms:", elapsed_time_ms * 1000)
    print(f"Avg Forward Time per Image: {avg_forward_time} ms")


if __name__ == "__main__":
    calcGPUTime()
