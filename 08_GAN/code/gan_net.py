
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import random
import time
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# folder path
DATA_PATH = './data'
MODEL_PATH = './model'
RESULTS_PATH = './results'

# random seed np/torch
seed = 42
random.seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

# hyperparameters
batch_size = 1
num_epochs = 1
lr = 1e-4
momentum = 0.5


# Create folder if not exist
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Download Fashion MNIST dataset
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((32, 32)),
                                transforms.Normalize((0.5,), (0.5,))])

# Download and load the training data
trainset = datasets.FashionMNIST(DATA_PATH, download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST(DATA_PATH, download=True, train=False, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
# Define the Generator and Discriminator networks
class Generator(nn.Module):
    """ Generator generates fake images from random noise."""
    def __init__(self):
        super(Generator, self).__init__()
        # Input is 100, going into a convolution.
        self.conv1 = nn.utils.spectral_norm(nn.ConvTranspose2d(100, 512, (4, 4), (1, 1), (0, 0), bias=False))
        self.bn1 = nn.BatchNorm2d(512)
        # state size. 512 x 4 x 4
        self.conv2 = nn.utils.spectral_norm(nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1), bias=False))
        self.bn2 = nn.BatchNorm2d(256)
         # state size. 256 x 8 x 8
        self.conv3 = nn.utils.spectral_norm(nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1), bias=False))
        self.bn3 = nn.BatchNorm2d(128)
        # state size. 128 x 16 x 16
        self.conv4 = nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        # state size. 64 x 32 x 32
        self.conv5 = nn.ConvTranspose2d(64, 1, (4, 4), (1, 1), (0, 0), bias=False)
        # state size. 1 x 32 x 32

    def forward(self, z):
        z = z.view(-1, 100, 1, 1)
        x = F.leaky_relu(self.bn1(self.conv1(z)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = torch.tanh(self.conv5(x))
        return x

class Discriminator(nn.Module):
    """ Discriminator learns to distinguish between real and fake images."""
    def __init__(self):
        super(Discriminator, self).__init__()
        # Input is 1 x 32 x 32
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(1, 64, (4, 4), (2, 2), (1, 1), bias=False))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(64, 128, (4, 4), (2, 2), (1, 1), bias=False))
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(128, 256, (4, 4), (2, 2), (1, 1), bias=False))
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(256, 1, (4, 4), (1, 1), (0, 0), bias=False))

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = self.conv4(x)
        x = x.view(-1, 1)
        x = nn.functional.sigmoid(x)
        return x

class DCGAN(nn.Module):
    """ DCGAN combines a generator and discriminator."""
    def __init__(self, latent_dim=100, img_size=(32,32), lr=1e-4, betas=(0.5, 0.999), device=None):
        super(DCGAN, self).__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._device = device
        self._lr = lr
        self._betas = betas
        self._latent_dim = latent_dim
        self._img_size = img_size

        # Initialize generator and discriminator
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)

        # Initialize weights
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)

        # Initialize optimizers and criterion
        self.criteria = nn.BCELoss()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self._lr, betas=self._betas)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self._lr, betas=self._betas)

    def forward(self, z):
        return self.generator(z)

    def sample_random_z(self, n):
        sample = torch.randn(n, self._latent_dim, 1, 1, device=self._device)
        return sample

    def sample_G(self, n):
        z = self.sample_random_z(n)
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        # binary cross-entropy loss
        return self.criteria(y_hat, y)

    def generator_loss(self, fake):
        return self.adversarial_loss(self.discriminator(fake), torch.ones(fake.size(0), 1, device=self._device))

    def discriminator_loss(self, real, fake):
        real_loss = self.adversarial_loss(self.discriminator(real), torch.ones(real.size(0), 1, device=self._device))
        fake_loss = self.adversarial_loss(self.discriminator(fake.detach()), torch.zeros(fake.size(0), 1, device=self._device))
        return real_loss + fake_loss

    def train_step(self, real_images):
        # ============================================================= #
        # ================== Train the discriminator ================== #
        # ============================================================= #
        # # (1) Update Discriminator network: maximize log(D(x)) + log(1 - D(G(z)))

        # Train the discriminator with real images
        self.optimizer_D.zero_grad()
        real_loss = self.adversarial_loss(self.discriminator(real_images), torch.ones(real_images.size(0), 1, device=self._device))

        # Train the discriminator with fake images
        z = self.sample_random_z(real_images.size(0))
        fake = self.generator(z)
        fake_loss = self.adversarial_loss(self.discriminator(fake.detach()), torch.zeros(fake.size(0), 1, device=self._device))
        
        # Total Discriminator loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.optimizer_D.step()

        # ============================================================= #
        # ===================== Train the generator =================== #
        # ============================================================= #
        # (2) Update Generator network: maximize log(D(G(z)))
        
        # Train the generator with random noise z
        self.optimizer_G.zero_grad()
        z = self.sample_random_z(real_images.size(0))
        fake = self.generator(z)

        # Generator loss
        g_loss = self.adversarial_loss(self.discriminator(fake), torch.ones(fake.size(0), 1, device=self._device))
        g_loss.backward()
        self.optimizer_G.step()

        return d_loss, g_loss

    # custom weights initialization called on ``netG`` and ``netD``
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def plot(self, n=6, save=False, path='results/result.png'):
        import matplotlib.pyplot as plt
        import numpy as np
        with torch.no_grad():
            z = self.sample_random_z(n)
            fake = self.generator(z).cpu()
            fake = fake.numpy()
            fig, axes = plt.subplots(5, 5, figsize=(10, 10))
            for i, ax in enumerate(axes.flat):
                ax.imshow(fake[i].reshape(32, 32), cmap='gray')
                ax.axis('off')
            if save:
                plt.savefig(path)
            plt.show()    

# Starting point of script
if __name__ == "main":
    # Initialize DCGAN
    dcgan = DCGAN(latent_dim=100, img_size=(32, 32), lr=1e-4, betas=(0.5, 0.999), device=device)

    d_training_losses = []
    g_training_losses = []

    # Train DCGAN
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(trainloader, 0):
            images = images.to(device)

            # Train the DCGAN
            d_loss, g_loss = dcgan.train_step(images)

            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}'
                    .format(epoch, num_epochs, i+1, len(trainloader), d_loss.item(), g_loss.item()))

            d_training_losses.append(d_loss.item())
            g_training_losses.append(g_loss.item())


    # Plot some images generated by the DCGAN
    dcgan.plot(save=True, path=RESULTS_PATH + '/dcgan.png')

    # Save the model under MODEL_PATH
    dcgan.save(MODEL_PATH + '/dcgan.pth')
