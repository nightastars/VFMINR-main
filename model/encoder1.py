import torch.nn as nn
import torch


class rmse(nn.Module):
    def __init__(self,depth,mid_channels=64):
        super(rmse,self).__init__()
        self.resblock = nn.Conv2d(mid_channels,mid_channels,kernel_size=3,padding=1,stride=1)
        self.sig = nn.Sigmoid()
        self.conv =nn.Sequential(nn.Conv2d(mid_channels,mid_channels,kernel_size=1,padding=0,stride=1),nn.PReLU(mid_channels),
                                 nn.Conv2d(mid_channels,mid_channels,kernel_size=1,padding=0,stride=1))
        self.depth = str(depth)

    def forward(self,x):
        output = []
        output.append(x)
        size = len(self.depth)
        for i in range(size):
            out1 = self.resblock(output[i])
            out = nn.AdaptiveAvgPool2d((1,1))(out1)
            out = self.conv(out)
            out = self.sig(out)
            out = torch.mul(out, out1)
            out = out + output[(i)]
            output.append(out)
        return x + output[(size-1)]


class ResCNN(nn.Module):
    def __init__(self, depth=2, feature_dim=128):
        super(ResCNN, self).__init__()
        self.conv_start = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True)
        )

        self.rmse1 = rmse(depth=depth, mid_channels=64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True)
        )

        self.rmse2 = rmse(depth=depth, mid_channels=64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True)
        )

        self.conv_end = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=feature_dim, kernel_size=3, padding=3 // 2)
        )

    def forward(self, x):
        conv_start = self.conv_start(x)
        rmse1 = self.rmse1(conv_start)
        conv1 = self.conv1(rmse1) + x
        rmse2 = self.rmse2(conv1)
        conv2 = self.conv2(rmse2) + x
        conv_end = self.conv_end(conv2)
        return conv_end


class ResCNN1(nn.Module):
    def __init__(self, feature_dim=128):
        super(ResCNN1, self).__init__()
        self.conv_start = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True)
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            # nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            # nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            # nn.ReLU(inplace=True)
        )
        self.conv_end = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=feature_dim, kernel_size=3, padding=3 // 2),
        )

    def forward(self, x):
        in_block1 = self.conv_start(x)
        out_block1 = self.block1(in_block1)
        in_block2 = out_block1 + in_block1
        out_block2 = self.block2(in_block2)
        in_block3 = out_block2 + in_block2
        out_block3 = self.block3(in_block3)
        res_img = self.conv_end(out_block3 + in_block3)
        return x + res_img


if __name__ == '__main__':
    model = ResCNN()
    # defining the input
    x = torch.rand(1, 1, 667, 912)

    # forward pass
    y = model(x)

    print(y.shape)

