
import numpy as np
import os
import torch
from torch.optim import Adam
from torchvision import models
from  aux_ops import preprocess_image, recreate_image, save_image


# Initialize GPU if available
use_gpu = False
if torch.cuda.is_available():
    use_gpu = True
# Select device to work on.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Total Variation Loss
def total_variation_loss(img, weight):
     # Your code here
    pass

def visualise_layer_filter(model, layer_nmbr, filter_nmbr, num_optim_steps=26):

    # Generate a random image
    rand_img = np.uint8(np.random.uniform(low=120,
                                          high=190,
                                          size=(224, 224, 3)))

    # Process image and return variable
    processed_image = preprocess_image(rand_img, False)
    processed_image = torch.tensor(processed_image, device=device).float()
    processed_image.requires_grad = True
    # Define optimizer for the image
    optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-5)
    for i in range(1, num_optim_steps):
        optimizer.zero_grad()
        # Assign create image to a variable to move forward in the model
        x = processed_image
        for index, layer in enumerate(model):
            # Forward pass layer by layer
            x = layer(x)
            if index == layer_nmbr:
                # Only need to forward until the selected layer is reached
                # Now, x is the output of the selected layer
                break

        conv_output = x[0, filter_nmbr]
        # Loss function is the mean of the output of the selected layer/filter
        # We try to minimize the mean of the output of that specific filter
        loss = -torch.mean(conv_output)
        # You may need to add total variation loss later
        # loss_tv = total_variation_loss(processed_image, 500.)
        # loss = -torch.mean(conv_output) + loss_tv*1.

        # print(f'Step {i:05d}. Loss:{loss.data.cpu().numpy():0.2f}')
        # Compute gradients
        loss.backward()
        # Apply gradients
        optimizer.step()
        # Recreate image
        optimized_image = recreate_image(processed_image.cpu())

    return optimized_image


if __name__ == '__main__':
    layer_nmbr = 28
    filter_nmbr = 228

    # Fully connected layer is not needed
    model = models.vgg16(pretrained=True).features
    model.eval()
    # Fix model weights
    for param in model.parameters():
        param.requires_grad = False
    # Enable GPU
    if use_gpu:
        model.cuda()

    # use this output in some way
    visualise_layer_filter(model,
                           layer_nmbr=layer_nmbr,
                           filter_nmbr=filter_nmbr)
