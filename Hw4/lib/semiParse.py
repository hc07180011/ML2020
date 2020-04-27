import numpy as np
from keras.models import load_model

def semiParse(model_path='./model/model.h5', threshold=0.9):

    train_x = np.load('./data/train_x.npy', allow_pickle=True)
    train_y = np.load('./data/train_y.npy', allow_pickle=True)
    train_x_no_label = np.load('./data/train_x_no_label.npy', allow_pickle=True)

    prediction = load_model(model_path).predict(train_x_no_label)

    idx1 = np.rot90(prediction > threshold)
    idx2 = np.rot90(prediction < 1-threshold)
    train_x = np.vstack((train_x, train_x_no_label[idx1[0]==True], train_x_no_label[idx2[0]==True]))
    train_y = np.append(train_y, np.ones(train_x_no_label[idx1[0]==True].shape[0], dtype='int'))
    train_y = np.append(train_y, np.zeros(train_x_no_label[idx2[0]==True].shape[0], dtype='int'))

    np.save('./data/semi-train_x.npy', train_x)
    np.save('./data/semi-train_y.npy', train_y)
