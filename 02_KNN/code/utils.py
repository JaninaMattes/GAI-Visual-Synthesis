from math import sqrt
import numpy as np
from matplotlib import pyplot as plt, ticker
from random import randint
from skimage import exposure
import imutils
import cv2, os

def generate_colors(amount: int) -> list:
    """Create random colours."""
    colors = []
    for i in range(amount):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    return colors

def scatter_2D(x1, x2, y, cdict = {0: 'red', 1: 'blue'}, title='', path='./result/'):
    """Create scatter plot without line plot."""

    # if path does not exist, create it
    if not os.path.exists(path):
        os.makedirs(path)

    img_name = title.replace(' ', '_')
    img_path = path + img_name

    fig, ax = plt.subplots()
    for g in np.unique(y):
        ix = np.where(y == g)
        ax.scatter(x1[ix], x2[ix], c=cdict[g], marker='.', label=f'class {g}')
    ax.legend()
    #plt.xlim(-2,5, 2.5)
    #plt.ylim(-1.5, 1.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

    fig.savefig(img_path, dpi=fig.dpi)


def scatter_2D_linspace(x1, x2, y, cdict = {0: 'red', 1: 'blue'}, title='', path='./result/', line_x=np.linspace(0, 1), line_y=np.linspace(0, 1)):
    """Create scatter plot with line plot."""

    # if path does not exist, create it
    if not os.path.exists(path):
        os.makedirs(path)

    img_name = title.replace(' ', '_')
    img_path = path + img_name

    fig, ax = plt.subplots()

    # use meshgrid
    X, Y = np.meshgrid(line_x, line_y)
    Z1 = np.exp(X * Y)
    z = 100 * Z1 # 100 points
    z[:5, :5] = -1
    z = np.ma.masked_where(z <= 0, z)
    
    cs = plt.contourf(X, Y, z,
                    locator = ticker.LogLocator(),
                    cmap ="bone")
                     
    plt.colorbar(cs)

    for g in np.unique(y):
        ix = np.where(y == g)
        ax.scatter(x1[ix], x2[ix], c=cdict[g], marker='.', label=f'class {g}', zorder=1)
    ax.legend()
    
    # Plot the resulting regression line
    plt.plot(line_x, line_y, '-', color='r', zorder=1)

    #plt.xlim(-2,5, 2.5)
    #plt.ylim(-1.5, 1.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

    fig.savefig(img_path, dpi=fig.dpi)

def calc_rss(y, y_pred):
    """Calculate minimal RSS Error"""
    return np.sum(np.square(y-y_pred))

def calc_accuracy(y, ypred):
    """Calculate the accuracy score."""
    return np.mean(y==ypred)

def euclidean_distance(vec1, vec2):
    """ Distance metric to define the similarity of two vectors
        
        Calculate the Euclidean distance between two vectors.
        >> Euclidean Distance = sqrt(sum i to N (x1_i – x2_i)^2)
        As the data points are vectors the norm can be calculated.
    """
    # return np.sqrt(np.sum((vec1-vec2)**2))
    return np.linalg.norm(vec1 - vec2) # vectorized

def convert_img(image):
	image = image.reshape((8, 8)).astype("float64")
	image = exposure.rescale_intensity(image, out_range=(0, 255))
	image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)
	return image