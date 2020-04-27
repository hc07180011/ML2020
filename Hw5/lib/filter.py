import os, sys, cv2, itertools
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import load_model
from keras import backend as K
from skimage.segmentation import slic

def plotImageFiltersResult(model, arrayX, intChooseId, output_path):
    intImageHeight = 128
    intImageWidth = 128
    dictLayer = dict([layer.name, layer] for layer in model.layers)
    inputImage = model.input
    listLayerNames = [layer for layer in dictLayer.keys() if "activation" in layer or "conv2d" in layer][:8]
    listCollectLayers = [K.function([inputImage, K.learning_phase()], [dictLayer[name].output]) for name in listLayerNames]

    fig, axes = plt.subplots(5, 5)

    axes[0][0].axis('off')
    for cnt, ic in enumerate([4, 15, 17, 53]):
        axes[cnt+1][0].imshow(plotWhiteNoiseActivateFilters(model, output_path, ic))

    for cnt, ic in enumerate(intChooseId):
        fn = listCollectLayers[2]
        arrayPhoto = arrayX[ic].reshape(1, intImageWidth, intImageHeight, 3)
        listLayerImage = fn([arrayPhoto, 0])
        axes[0][cnt+1].imshow(arrayX[ic][...,::-1])
        axes[1][cnt+1].imshow(listLayerImage[0][0, :, :, 4] * 2.0)
        axes[2][cnt+1].imshow(listLayerImage[0][0, :, :, 15] * 2.0)
        axes[3][cnt+1].imshow(listLayerImage[0][0, :, :, 17] * 2.0)
        axes[4][cnt+1].imshow(listLayerImage[0][0, :, :, 53] * 2.0)

    plt.savefig(os.path.join(output_path, '5-2.png'))

def deprocessImage(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def makeNormalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def trainGradAscent(intIterationSteps, arrayInputImageData, targetFunction, intRecordFrequent):
    listFilterImages = []
    floatLearningRate = 1e-2
    for i in range(intIterationSteps):
        floatLossValue, arrayGradientsValue = targetFunction([arrayInputImageData, 0])
        arrayInputImageData += arrayGradientsValue * floatLearningRate
        if i % intRecordFrequent == 0:
            listFilterImages.append((arrayInputImageData, floatLossValue))
    return listFilterImages

def plotWhiteNoiseActivateFilters(model, output_path, i):
    intRecordFrequent = 20
    intNumberSteps = 160
    intIterationSteps = 160

    dictLayer = dict([layer.name, layer] for layer in model.layers)
    inputImage = model.input
    listLayerNames = [layer for layer in dictLayer.keys() if "activation" in layer or "conv2d" in layer]
    listCollectLayers = [dictLayer[name].output for name in listLayerNames]

    filter_num = [64, 128, 128, 256, 256, 512, 512]

    fn = listCollectLayers[2]
    listFilterImages = []
    intFilters = filter_num[2]
    arrayInputImage = np.random.random((1, 128, 128, 3))
    tensorTarget = K.mean(fn[:, :, :, i])

    tensorGradients = makeNormalize(K.gradients(tensorTarget, inputImage)[0])
    targetFunction = K.function([inputImage, K.learning_phase()], [tensorTarget, tensorGradients])

    listFilterImages = (trainGradAscent(intIterationSteps, arrayInputImage, targetFunction, intRecordFrequent))

    return deprocessImage(listFilterImages[0][0].squeeze())

def conv_filter(output_path, model, train_x, train_y):
    plotImageFiltersResult(model, train_x, [83, 4218, 4707, 8598], output_path)
