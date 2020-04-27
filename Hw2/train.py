import numpy as np
import sys

np.random.seed(0)
X_train_fpath = sys.argv[3]
Y_train_fpath = sys.argv[4]
X_test_fpath = sys.argv[5]
output_fpath = sys.argv[6]
bound = [0, 18, 25, 35, 45, 55, 65, 75, 100]

def _normalize(X, train=True, specified_column=None, X_mean=None, X_std=None):
    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)
    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
    return X, X_mean, X_std

def _train_dev_split(X, Y, dev_ratio=0.25):
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def _sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

def _f(X, w, b):
    return _sigmoid(np.matmul(X, w) + b)

def _predict(X, w, b):
    return np.round(_f(X, w, b)).astype(np.int)
    
def _accuracy(Y_pred, Y_label):
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

def _cross_entropy_loss(y_pred, Y_label):
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
    return cross_entropy

def _gradient(X, Y_label, w, b):
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad

with open(X_train_fpath) as f:
	next(f)
	X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
	for ic in range(len(bound)-1):
		buf = np.ones((X_train.shape[0], 1)); buf[bound[ic+1]<=(np.rot90(X_train)[-1])] = 0; buf[(np.rot90(X_train)[-1])<bound[ic]] = 0
		X_train = np.hstack((X_train, buf))
	X_train = np.delete(X_train, 0, 1)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype=float)
with open(X_test_fpath) as f:
	next(f)
	X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
	for ic in range(len(bound)-1):
		buf = np.ones((X_test.shape[0], 1)); buf[bound[ic+1]<=(np.rot90(X_test)[-1])] = 0; buf[(np.rot90(X_test)[-1])<bound[ic]] = 0
		X_test = np.hstack((X_test, buf))
	X_test = np.delete(X_test, 0, 1)

X_train, X_mean, X_std = _normalize(X_train, train=True)
X_test, _, _= _normalize(X_test, train=False, specified_column=None, X_mean=X_mean, X_std=X_std)

X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio=0.2)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
'''
print('Size of training set: {}'.format(train_size))
print('Size of development set: {}'.format(dev_size))
print('Size of testing set: {}'.format(test_size))
print('Dimension of data: {}'.format(data_dim))
'''
w = np.zeros((data_dim,)) 
b = np.zeros((1,))

max_iter = 250
batch_size = 16
learning_rate = 0.01

train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

step_w = 1e-8
step_b = 1e-8
for epoch in range(max_iter):
    X_train, Y_train = _shuffle(X_train, Y_train)
    for idx in range(int(np.floor(train_size / batch_size))):
        X = X_train[idx*batch_size:(idx+1)*batch_size]
        Y = Y_train[idx*batch_size:(idx+1)*batch_size]
        w_grad, b_grad = _gradient(X, Y, w, b)
        step_w += w_grad ** 2
        step_b += b_grad ** 2
        w = w - learning_rate/np.sqrt(step_w) * w_grad
        b = b - learning_rate/np.sqrt(step_b) * b_grad

    '''
    y_train_pred = _f(X_train, w, b)
    Y_train_pred = np.round(y_train_pred)
    train_acc.append(_accuracy(Y_train_pred, Y_train))
    train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)

    y_dev_pred = _f(X_dev, w, b)
    Y_dev_pred = np.round(y_dev_pred)
    dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
    dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)

    print('{}/{}:'.format(epoch+1, max_iter))
    print('Training loss: {}'.format(train_loss[-1]))
    print('Development loss: {}'.format(dev_loss[-1]))
    print('Training accuracy: {}'.format(train_acc[-1])) 
    print('Development accuracy: {}'.format(dev_acc[-1]))
    '''

'''
np.save('w', w)
np.save('b', b)

import matplotlib.pyplot as plt

plt.plot(train_loss)
plt.plot(dev_loss)
plt.title('Loss')
plt.legend(['train', 'dev'])
plt.savefig('loss.png')
plt.show()

plt.plot(train_acc)
plt.plot(dev_acc)
plt.title('Accuracy')
plt.legend(['train', 'dev'])
plt.savefig('acc.png')
plt.show()
'''
predictions = _predict(X_test, w, b)
with open(output_fpath, 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))
'''
ind = np.argsort(np.abs(w))[::-1]
with open(X_test_fpath) as f:
    content = f.readline().strip('\n').split(',')
features = np.array(content)
for i in ind[0:10]:
    print(i, features[i+1], w[i])
'''