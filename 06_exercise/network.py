import torch 
import torch.nn as nn
import torch.nn.functional as F

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
    return x + (noise * factor)


class Encoder(nn.Module):
    """Encoder network for the Autoencoder"""

    def __init__(self, input_shape=(28, 28)):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_shape[0] * input_shape[1], 128),
            nn.LeakyReLU(0.1), # selected Leaky ReLU activation function
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 8),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    """Decoder network for the Autoencoder"""

    def __init__(self, input_shape=(28, 28)):
        super(Decoder, self).__init__()
        self.input_shape = input_shape
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 784),
            nn.LeakyReLU(0.1),
            nn.Linear(784, input_shape[0] * input_shape[1]),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.decoder(x)
        x = x.view(x.size(0), 1, self.input_shape[0], self.input_shape[1])
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
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(4), # added batch normalization
            nn.Conv2d(4, 8, kernel_size=5),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(8),
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
            nn.LeakyReLU(0.1),
            nn.Linear(400, 4000),
            nn.LeakyReLU(0.1),
            nn.Unflatten(1, (10, 20, 20)),
            nn.ConvTranspose2d(10, 10, kernel_size=5),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(10, 1, kernel_size=5),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

class ConvAutoencoder(nn.Module):
    """ Convolutional Autoencoder network"""

    def __init__(self, input_shape=(28, 28)):
        super(ConvAutoencoder, self).__init__()
        self.input_shape = input_shape
        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()

    def forward(self, x):
        x = x.view(x.size(0), 1, self.input_shape[0], self.input_shape[1])
        z = self.encoder(x)
        x = self.decoder(z)
        return x.view(-1, 1, self.input_shape[0], self.input_shape[1])
    

class VanillaVAEEncoder(nn.Module):

    def __init__(self):
        super(VanillaVAEEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1),
            nn.Conv2d(4, 8, kernel_size=5),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
            nn.Linear(3200, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class VanillaVAEDecoder(nn.Module):

    def __init__(self):
        super(VanillaVAEDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(10, 400),
            nn.LeakyReLU(0.1),
            nn.Linear(400, 4000),
            nn.LeakyReLU(0.1),
            nn.Unflatten(1, (10, 20, 20)),
            nn.ConvTranspose2d(10, 10, kernel_size=5),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(10, 1, kernel_size=5),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x
    


class VanillaVAE(nn.Module):
    """ Variational Autoencoder network"""

    def __init__(self, latent_dim=2):
        super(VanillaVAE, self).__init__()
        self.latent_dim = latent_dim
        # 1. Encoder
        self.encoder = VanillaVAEEncoder()
        # latent mean and variance
        self.fc_mean = nn.Linear(10, latent_dim)
        self.fc_logvar = nn.Linear(10, latent_dim)
        # 2. Decoder
        self.decoder = VanillaVAEDecoder()

    def reparameterize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mean)
    
    def loss_function(self, recon_x, input, mean, logvar):
        # reconstruction loss
        BCE = F.mse_loss(recon_x, input, size_average=False)
        # KL divergence loss
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return BCE + KLD

    def forward(self, x):
        x = x.view(x.size(0), 1, self.input_shape[0], self.input_shape[1])
        z = self.encoder(x) 

        # encodes the input into the latent space code
        # -- split the tensor into mu and logvar
        mean = self.fc_mean(z)
        logvar = self.fc_logvar(z)
        z = self.reparameterize(mean, logvar)
        x = self.decoder(z)
        x = x.view(-1, 1, self.input_shape[0], self.input_shape[1])
        return x, mean, logvar
