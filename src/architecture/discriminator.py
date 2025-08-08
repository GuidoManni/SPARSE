import torch
import torch.nn as nn


# discriminator.py
class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), c_dim=5, n_strided=6):
        super(Discriminator, self).__init__()
        channels, img_size, _ = img_shape
        self.c_dim = c_dim

        def discriminator_block(in_filter, out_filters):
            layers = [
                nn.Conv2d(in_filter, out_filters, 4, stride=2, padding=1),
                nn.LeakyReLU(0.01)
            ]
            return layers

        layers = discriminator_block(channels, 64)
        curr_dim = 64

        for _ in range(n_strided-1):
            layers.extend(discriminator_block(curr_dim, curr_dim*2))
            curr_dim = curr_dim*2

        self.model = nn.Sequential(*layers)

        # Two outputs now:
        # 1. PatchGAN (real/fake)
        self.out1 = nn.Conv2d(curr_dim, 1, 3, padding=1, bias=False)
        # 2. Class Prediction
        kernel_size = img_size // (2 ** n_strided)
        self.out2 = nn.Conv2d(curr_dim, c_dim, kernel_size, bias=False)

    def forward(self, img):
        feature_repr = self.model(img)
        out_adv = self.out1(feature_repr)  # PatchGAN score
        out_cls = self.out2(feature_repr)  # class prediction

        return out_adv, out_cls.view(out_cls.size(0), -1)
