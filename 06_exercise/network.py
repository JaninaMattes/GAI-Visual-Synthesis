import torch.nn as nn


def to_img(x):
    """ Maps a 2D tensor from range [-1, 1] to 4D tensor with range [0, 1].
    Useful for plotting of reconstructions.

    :param x: 2D Tensor that is supposed to be converted
    :return: Converted 4D Tensor with b, c, w, h, where w = h = 28
    """
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


def add_white_noise(x, factor=0.5, stddev=1):
    """ Adds white noise to an input tensor.
    To make sure that data is in intended range [min, max], use torch.clamp(x, min, max) after applying this function.

    :param x: ND Tensor that is altered
    :param factor: A factor that controls the strength of the additive noise
    :param stddev: The stddev of the normal distribution used for generating the noise
    :return: ND Tensor, x with white noise
    """
    # add white noise to tensor
    noise = x.clone().normal_(0, stddev)
    return x + noise * factor


class Encoder(nn.Module):
    """Encoder network for the Autoencoder"""

    def __init__(self, input_shape=(28, 28)):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_shape[0] * input_shape[1], 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16, 8),
            nn.LeakyReLU(True)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    """Decoder network for the Autoencoder"""

    def __init__(self, input_shape=(28, 28)):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16, 32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.decoder(x)
        x = x.view(x.size(0), 28, 28)
        return x


class Autoencoder(nn.Module):
    """ Autoencoder network"""

    def __init__(self, input_shape=(28, 28)):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_shape)
        self.decoder = Decoder(input_shape)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x

class ConvEncoder(nn.Module):
    """ Convolutional Encoder network for the Autoencoder"""

    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(4, 8, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(3200, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
    
class ConvDecoder(nn.Module):
    """ Convolutional Decoder network for the Autoencoder"""

    def __init__(self):
        super(ConvDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(10, 400),
            nn.LeakyReLU(inplace=True),
            nn.Linear(400, 4000),
            nn.LeakyReLU(inplace=True),
            nn.Unflatten(1, (10, 20, 20)),
            nn.ConvTranspose2d(10, 10, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(10, 1, kernel_size=5),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

class ConvAutoencoder(nn.Module):
    """ Convolutional Autoencoder network"""

    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        z = self.encoder(x)
        x = self.decoder(z)
        return x.view(-1, 1, 28, 28)