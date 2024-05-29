
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
    # Calculate total variation loss
    return weight * (torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])))

def visualise_layer_filter(model, layer_nmbr, filter_nmbr, num_optim_steps=26, lr=0.1, optimizer_type='adam', rand_img=None, total_var_loss=False):

    # Generate a random image
    if rand_img is None:
        rand_img = np.uint8(np.random.uniform(low=120,
                                            high=190,
                                            size=(224, 224, 3)))

    # Process image and return variable
    processed_image = preprocess_image(rand_img, False)
    processed_image = torch.tensor(processed_image, device=device).float()
    processed_image.requires_grad = True
    # Define optimizer for the image
    optimizer = get_optimizer(optimizer_type, processed_image, lr=lr)
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

        if total_var_loss:
            # Add total variation loss later
            loss_tv = total_variation_loss(processed_image, 500.)
            loss = -torch.mean(conv_output) + (loss_tv*1.)
        
        else:
            # Mean of the output of the selected layer/filter
            loss = -torch.mean(conv_output)

        if i % 20 == 0:
            print(f'Step {i:05d}. Loss:{loss.data.cpu().numpy():0.2f}')
        
        # Compute gradients
        loss.backward()
        # Apply gradients
        optimizer.step()
        # Recreate image
        optimized_image = recreate_image(processed_image.cpu())

    return optimized_image

def get_optimizer(optimizer_type, input_img, lr=0.1, weight_decay=0.0):
    """
    Get optimizer for the input image.

    Args:
    - optimizer_type (str): The type of optimizer to use. Must be one of 'adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', or 'adamax'.
    - input_img (torch.Tensor): The input image tensor to optimize.
    - lr (float): The learning rate for the optimizer.
    - weight_decay (float, optional): The weight decay (L2 penalty) for the optimizer. Defaults to 0.

    Returns:
    - optimizer (torch.optim.Optimizer): The selected optimizer instance.
    """

    # Define a mapping of optimizer names to their corresponding classes and default weight decay values
    optimizer_configs = {
        'adam': (torch.optim.Adam, 1e-5),
        'sgd': (torch.optim.SGD, 1e-6),
        'asgd': (torch.optim.ASGD, 1e-6), 
        'rmsprop': (torch.optim.RMSprop, 1e-5),
        'adagrad': (torch.optim.Adagrad, 1e-5),
        'adadelta': (torch.optim.Adadelta, 1e-5),
        'adamax': (torch.optim.Adamax, 1e-5),
        'adamw': (torch.optim.AdamW, 1e-4),
        'sparseadam': (torch.optim.SparseAdam, 1e-5),
    }

    # Check if the selected optimizer is valid
    if optimizer_type not in optimizer_configs:
        raise ValueError('[Info] Please select a valid optimizer type: ' + ', '.join(optimizer_configs.keys()))

    # Get the optimizer class and default weight decay value for the selected optimizer
    optimizer_class, default_weight_decay = optimizer_configs[optimizer_type]

    # Create an instance of the selected optimizer with the correct weight decay value
    optimizer = optimizer_class([input_img], lr=lr, weight_decay=default_weight_decay if weight_decay == 0 else weight_decay)

    return optimizer



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
