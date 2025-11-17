import torch
import torch.nn as nn

# --- 1. Baseline DCGAN Models (for 32x32x3 CIFAR-10) ---

class Generator_CIFAR(nn.Module):
    def __init__(self, noise_dim, channels_img, features_g):
        super(Generator_CIFAR, self).__init__()
        # Input: N x noise_dim x 1 x 1
        self.net = nn.Sequential(
            self._block(noise_dim, features_g * 4, 4, 1, 0),      # N x f_g*4 x 4 x 4
            self._block(features_g * 4, features_g * 2, 4, 2, 1), # N x f_g*2 x 8 x 8
            self._block(features_g * 2, features_g, 4, 2, 1),     # N x f_g x 16 x 16
            nn.ConvTranspose2d(
                features_g, channels_img, 4, 2, 1, bias=False
            ), # N x channels_img x 32 x 32
            nn.Tanh() # Output [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator_CIFAR(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator_CIFAR, self).__init__()
        # Input: N x channels_img x 32 x 32
        self.net = nn.Sequential(
            nn.Conv2d(channels_img, features_d, 4, 2, 1, bias=False), # N x f_d x 16 x 16
            nn.LeakyReLU(0.2, inplace=True),
            
            self._block(features_d, features_d * 2, 4, 2, 1),      # N x f_d*2 x 8 x 8
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # N x f_d*4 x 4 x 4
            
            nn.Conv2d(features_d * 4, 1, 4, 1, 0, bias=False),   # N x 1 x 1 x 1
            nn.Sigmoid() # Final probability
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)


# --- 2. Improved WGAN-GP Models (for 32x32x3 CIFAR-10) ---

class WGANGP_Generator_CIFAR(nn.Module):
    def __init__(self, noise_dim, channels_img, features_g):
        super(WGANGP_Generator_CIFAR, self).__init__()
        # Input: N x noise_dim x 1 x 1
        self.net = nn.Sequential(
            self._block(noise_dim, features_g * 4, 4, 1, 0),      # N x f_g*4 x 4 x 4
            self._block(features_g * 4, features_g * 2, 4, 2, 1), # N x f_g*2 x 8 x 8
            self._block(features_g * 2, features_g, 4, 2, 1),     # N x f_g x 16 x 16
            nn.ConvTranspose2d(
                features_g, channels_img, 4, 2, 1, bias=False
            ), # N x channels_img x 32 x 32
            nn.Tanh() # Output [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)

class WGANGP_Critic_CIFAR(nn.Module):
    def __init__(self, channels_img, features_d):
        super(WGANGP_Critic_CIFAR, self).__init__()
        # Input: N x channels_img x 32 x 32
        self.net = nn.Sequential(
            nn.Conv2d(channels_img, features_d, 4, 2, 1, bias=False), # N x f_d x 16 x 16
            nn.LeakyReLU(0.2, inplace=True),
            
            self._block(features_d, features_d * 2, 4, 2, 1),      # N x f_d*2 x 8 x 8
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # N x f_d*4 x 4 x 4
            
            nn.Conv2d(features_d * 4, 1, 4, 1, 0, bias=False),   # N x 1 x 1 x 1
            # NO SIGMOID
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            # WGAN-GP paper recommends NOT using BatchNorm in Critic
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)


# --- 3. Utility Function ---

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)