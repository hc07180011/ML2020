import os, sys, cv2
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import slic

np.random.seed(0)

def plotLime(output_path, model, train_x, train_y):

    def segmentation(data):
        return slic(data, n_segments=100, compactness=1, sigma=1)

    images, labels = [train_x[83][...,::-1], train_x[4218], train_x[4707][...,::-1], train_x[8598][...,::-1]], train_y[[83, 4218, 4707, 8598]]
    fig, axs = plt.subplots(1, 4)

    for idx, (image, label) in enumerate(zip(images, labels)):
        explainer = lime_image.LimeImageExplainer()
        explaination = explainer.explain_instance(image=image, hide_color=None, classifier_fn=model.predict_on_batch, segmentation_fn=segmentation)
        lime_img, mask = explaination.get_image_and_mask(
                                    label=explaination.top_labels[0],
                                    positive_only=False,
                                    hide_rest=False,
                                    num_features=11,
                                    min_weight=0.05)
        axs[idx].imshow(lime_img[...,::-1] if idx == 1 else lime_img)

    plt.savefig(os.path.join(output_path, '5-3.png'))
    plt.close()
