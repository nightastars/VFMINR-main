import torch
import torch.nn as nn
from model.encoder import ResCNN1, ResCNN2
from model.mlp import MLP
from utils import make_coord
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, feature_dim, encoder_depth, decoder_depth, decoder_width):
        super(Network, self).__init__()
        # self.encoder1 = ResCNN1(depth=encoder_depth, feature_dim=feature_dim)
        # self.encoder2 = ResCNN2(depth=encoder_depth//2, feature_dim=64)
        self.decoder = MLP(in_dim=feature_dim+42, out_dim=1, depth=decoder_depth, width=decoder_width)
        # self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Sequential(nn.Conv2d(128+64, feature_dim, kernel_size=5, padding=5//2, stride=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=3//2, stride=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=3//2, stride=1))

    def forward(self, xy_input, xy_sinogram, feature):   # sinogram1, xy_input_pe, feature
        # extract feature map from sinogram image
        # print(xy_sinogram.size())
        feature_map = self.conv(feature)
        # print(feature_map.size())

        # print(feature_map.shape)
        # generate feature vector for coordinate through trilinear interpolation (Equ. 4 & Fig. 3).
        feature_vector = F.grid_sample(feature_map, xy_input.flip(-1).unsqueeze(1),
                                       mode='bilinear',
                                       align_corners=False)[:, :, 0, :].permute(0, 2, 1)
        # print(feature_vector.size())
        # concatenate coordinate with feature vector
        feature_vector_and_xyz_hr = torch.cat([feature_vector, xy_sinogram], dim=-1)
        # estimate the voxel intensity at the coordinate by using decoder.
        N, K = xy_sinogram.shape[:2]
        intensity_pre = self.decoder(feature_vector_and_xyz_hr.view(N * K, -1)).view(N, K, -1)
        return intensity_pre




if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # images = torch.randn((1, 1, 90, 729)) # .to(device)
    image1 = torch.randn((1, 1, 200, 921))
    image2 = torch.randn((1, 184200, 2))
    model = Network(feature_dim=192, encoder_depth=8, decoder_depth=8, decoder_width=258)
    output = model(image1, image2)
    print(model)
    print(output.shape)

    # vit = ResCNN(feature_dim=128) # .to(device)
    # output = vit(images)
    # print(output.shape)


