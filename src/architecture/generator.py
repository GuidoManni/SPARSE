import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p


class attention_gate(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        return out * s


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.ag = attention_gate(in_c, out_c)
        self.c1 = conv_block(in_c[0] + out_c, out_c)

    def forward(self, x, s):
        x = self.up(x)
        s = self.ag(x, s)
        x = torch.cat([x, s], axis=1)
        x = self.c1(x)
        return x


class attention_unet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.e1_gen = encoder_block(3 + num_classes, 64)
        self.e1_cls = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)

        self.b1 = conv_block(256, 512)

        self.d1 = decoder_block([512, 256], 256)
        self.d2 = decoder_block([256, 128], 128)
        self.d3 = decoder_block([128, 64], 64)

        self.output = nn.Conv2d(64, 3, kernel_size=1, padding=0)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, z=None, mode="generator"):
        if mode == "generator":
            batch_size, _, height, width = x.size()
            z_c_expanded = z.view(batch_size, self.num_classes, 1, 1).expand(batch_size, self.num_classes, height, width)
            x = torch.cat([x, z_c_expanded], axis=1)
            s1, p1 = self.e1_gen(x)
        else:
            s1, p1 = self.e1_cls(x)

        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        b1 = self.b1(p3)

        if mode == "classifier":
            class_pred = self.classifier(b1)
            return class_pred

        d1 = self.d1(b1, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)
        output = self.output(d3)
        return output


def sample_z(batch_size, num_continuous, num_classes):
    # Sample z_n from a Gaussian distribution
    z_n = torch.randn(batch_size, num_continuous)

    # Sample z_c as one-hot encoded vectors
    indices = torch.randint(low=0, high=num_classes, size=(batch_size,))
    z_c = F.one_hot(indices, num_classes=num_classes).float()

    # Concatenate z_n and z_c to form the latent variable z
    z = torch.cat((z_n, z_c), dim=1)

    return z, z_n, z_c, indices

# Example usage
if __name__ == "__main__":
    batch_size = 4
    num_continuous = 10
    num_classes = 5
    input_shape = (3, 128, 128)  # Example input shape (channels, height, width)

    x = torch.randn(batch_size, *input_shape)
    z, z_n, z_c, indices = sample_z(batch_size, num_continuous, num_classes)

    model = attention_unet(num_classes=num_classes)
    output = model(x, z_c, mode="classifier")

    print(output.shape)
