import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, GRU, Dropout, Dense, Conv1D, MaxPooling1D, SpatialDropout1D, LSTM
from keras.callbacks import ModelCheckpoint, Callback
from keras.regularizers import l2
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from lib.callback import make_table

def train(model_path='./model/model.h5', batch_size=256, epochs=5, test_size=0.2, seed=900104, new_model=True):

    np.random.seed(seed)
    train_x = np.load('./data/train_x.npy', allow_pickle=True)
    train_y = np.load('./data/train_y.npy', allow_pickle=True)
    embedding_matrix = np.load('./data/embedding_matrix.npy', allow_pickle=True)
    num, dim = embedding_matrix.shape
    train_acc = []
    val_acc = []
    with open('./data/progress.txt', 'w') as fp:
        fp.close()

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=test_size, random_state=np.random.randint(seed), shuffle=True)

    if new_model:
        model = Sequential()
        model.add(Embedding(num, dim, weights=[embedding_matrix], input_length=train_x.shape[1], trainable=False))
        model.add(Bidirectional(GRU(128, return_sequences=True), merge_mode='concat'))
        model.add(Bidirectional(GRU(128), merge_mode='concat'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        model.summary()

    else:
        model = load_model(model_path)

    checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    class MyCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            train_acc.append("{:.4f}".format(logs['acc']))
            val_acc.append("{:.4f}".format(logs['val_acc']))
            make_table(epoch, epochs, train_acc, val_acc)

    history = model.fit(train_x,
                        train_y,
                        validation_data=(val_x, val_y),
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[checkpoint, MyCallback()])

    return history
