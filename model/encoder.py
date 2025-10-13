import torch.nn as nn
import torch


class MS_CAM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, inter_channels, kernel_size=5, stride=1, padding=5 // 2),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, inter_channels, kernel_size=5, stride=1, padding=5 // 2),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei


class ResCNN1(nn.Module):
    def __init__(self, depth=8, feature_dim=128):
        super(ResCNN1, self).__init__()
        cam = []

        self.conv_start = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=5 // 2),
            nn.ReLU(inplace=True)
        )

        for _ in range(depth):
            cam.append(MS_CAM(channels=64, r=4))
            cam.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2))
            cam.append(nn.ReLU(inplace=True))

        self.res = nn.Sequential(*cam)

        self.conv_end = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=feature_dim, kernel_size=3, padding=3 // 2)
        )

    def forward(self, x):
        conv_start = self.conv_start(x)
        # print(conv_start.shape)
        cam = self.res(conv_start)
        conv_end = self.conv_end(cam)
        return conv_end


class ResCNN2(nn.Module):
    def __init__(self, depth=8, feature_dim=128):
        super(ResCNN2, self).__init__()
        cam = []

        self.conv_start = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=5 // 2),
            nn.ReLU(inplace=True)
        )

        for _ in range(depth):
            cam.append(MS_CAM(channels=64, r=4))
            cam.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2))
            cam.append(nn.ReLU(inplace=True))

        self.res = nn.Sequential(*cam)

        self.conv_end = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=feature_dim, kernel_size=3, padding=3 // 2)
        )

    def forward(self, x):
        conv_start = self.conv_start(x)
        # print(conv_start.shape)
        cam = self.res(conv_start)
        conv_end = self.conv_end(cam)
        return conv_end



# 代码参考 “Attentional feature fusion” https://cloud.tencent.com/developer/article/1895776
# class MS_CAM(nn.Module):
#     '''
#     单特征 进行通道加权,作用类似SE模块
#     '''
#
#     def __init__(self, channels=64, r=4):
#         super(MS_CAM, self).__init__()
#         inter_channels = int(channels // r)
#
#         self.local_att = nn.Sequential(
#             nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
#         )
#
#         self.global_att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
#         )
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         xl = self.local_att(x)
#         xg = self.global_att(x)
#         xlg = xl + xg
#         wei = self.sigmoid(xlg)
#         return x * wei


# class AFF(nn.Module):
#     '''
#     多特征融合 AFF
#     '''
#
#     def __init__(self, channels=64, r=4):
#         super(AFF, self).__init__()
#         inter_channels = int(channels // r)
#
#         self.local_att = nn.Sequential(
#             nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
#         )
#
#         self.global_att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
#         )
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x, residual):
#         xa = x + residual
#         xl = self.local_att(xa)
#         xg = self.global_att(xa)
#         xlg = xl + xg
#         wei = self.sigmoid(xlg)
#
#         xo = 2 * x * wei + 2 * residual * (1 - wei)
#         return xo


# class iAFF(nn.Module):
#     '''
#     多特征融合 iAFF
#     '''
#
#     def __init__(self, channels=64, r=4):
#         super(iAFF, self).__init__()
#         inter_channels = int(channels // r)
#
#         # 本地注意力
#         self.local_att = nn.Sequential(
#             nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
#         )
#
#         # 全局注意力
#         self.global_att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
#         )
#
#         # 第二次本地注意力
#         self.local_att2 = nn.Sequential(
#             nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
#         )
#         # 第二次全局注意力
#         self.global_att2 = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
#         )
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x, residual):
#         xa = x + residual
#         xl = self.local_att(xa)
#         xg = self.global_att(xa)
#         xlg = xl + xg
#         wei = self.sigmoid(xlg)
#         xi = x * wei + residual * (1 - wei)
#
#         xl2 = self.local_att2(xi)
#         xg2 = self.global_att(xi)
#         xlg2 = xl2 + xg2
#         wei2 = self.sigmoid(xlg2)
#         xo = x * wei2 + residual * (1 - wei2)
#         return xo


if __name__ == '__main__':
    model = ResCNN1()
    # defining the input
    x = torch.rand(1, 1, 667, 912)

    # forward pass
    y = model(x)

    print(y.shape)

