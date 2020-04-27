import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.activations import linear
from vis.utils import utils
from vis.visualization import visualize_saliency

def saliency(output_path, model, train_x, train_y):

    model.layers[23].activation = linear
    model = utils.apply_modifications(model)

    indices_to_visualize = [83, 4218, 4707, 8598]

    cnt = 0
    fig, axes = plt.subplots(2, 4)
    for index_to_visualize in indices_to_visualize:
        input_image = train_x[index_to_visualize]
        input_class = np.argmax(train_y[index_to_visualize])
        visualization = visualize_saliency(model, 23, filter_indices=input_class, seed_input=input_image)
        axes[0][cnt].imshow(input_image[...,::-1])
        axes[0][cnt].set_title('Original image')
        axes[1][cnt].imshow(visualization)
        axes[1][cnt].set_title('Saliency map')
        cnt += 1
    plt.savefig(os.path.join(output_path, '5-1.png'))
    plt.close()
