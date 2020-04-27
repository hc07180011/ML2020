import os
import sys
import numpy as np
import torch
import gensim
from gensim.models import word2vec
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from keras.utils.np_utils import *

sen_len = 32
vec_size = 300

def load_training_data(path='training_label.txt'):
    if 'training_label' in path:
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]
        x = [line[2:] for line in lines]
        y = [line[0] for line in lines]
        return x, y
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        return x

def load_testing_data(path='testing_data'):
    with open(path, 'r') as f:
        lines = f.readlines()
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        X = [sen.split(' ') for sen in X]
    return X

def train_word2vec(x):
    model = word2vec.Word2Vec(x, size=vec_size, window=5, min_count=5, workers=12, iter=10, sg=1)
    return model

def embedding_x(word2idx, data):
    sentence_list = []
    for i, sen in enumerate(data):
        sentence_idx = []
        for word in sen:
            if (word in word2idx.keys()):
                sentence_idx.append(word2idx[word])
            else:
                sentence_idx.append(word2idx["<UNK>"])
        if len(sentence_idx) > sen_len:
            sentence_idx = sentence_idx[:sen_len]
        else:
            pad_len = sen_len - len(sentence_idx)
            for _ in range(pad_len):
                sentence_idx.append(word2idx["<PAD>"])
        sentence_list.append(sentence_idx)
    return sentence_list

def do_preprocess(train_path='./data/training_label.txt',
                  train_nolabel_path='./data/training_nolabel.txt',
                  test_path='./data/testing_data.txt',
                  test_exist=False,
                  save_w2v=False):

    print('Reading files...')
    train_x, y = load_training_data(train_path)
    train_x_no_label = load_training_data(train_nolabel_path)
    if test_exist:
        test_x = load_testing_data(test_path)
    with open('./data/word2idx.txt', 'r') as fp:
        buf = fp.read()
        fp.close()
    word2idx = eval(buf)

    '''
    print('Training the model...')
    if test_exist:
        w2v_model = train_word2vec(train_x + train_x_no_label + test_x)
    else:
        w2v_model = train_word2vec(train_x + train_x_no_label)
    if save_w2v:
        w2v_model.save('./model/w2v.model')

    print('Embedding words...')
    idx2word = []
    word2idx = {}
    embedding_matrix = np.zeros(vec_size)
    for i, word in enumerate(w2v_model.wv.vocab):
        print('embedding: ', i)
        word2idx[word] = len(word2idx)
        idx2word.append(word)
        embedding_matrix = np.vstack((embedding_matrix, np.array(w2v_model[word])))
    word2idx['<PAD>'] = len(word2idx)
    idx2word.append('<PAD>')
    embedding_matrix = np.vstack((embedding_matrix, np.array(torch.empty(1, w2v_model.vector_size))))
    word2idx['<UNK>'] = len(word2idx)
    idx2word.append('<UNK>')
    embedding_matrix = np.vstack((embedding_matrix, np.array(torch.empty(1, w2v_model.vector_size))))
    embedding_matrix = np.delete(embedding_matrix, 0, 0)
    '''

    train_x = np.array(embedding_x(word2idx, train_x))
    train_y = np.array(y)
    train_x_no_label = np.array(embedding_x(word2idx, train_x_no_label))
    if test_exist:
        test_x = np.array(embedding_x(word2idx, test_x))

    print('save...')
    np.save('./data/train_x', train_x)
    np.save('./data/train_y', train_y)
    np.save('./data/train_x_no_label', train_x_no_label)
    if test_exist:
        np.save('./data/test_x', test_x)
    #np.save('./data/embedding_matrix', embedding_matrix)

