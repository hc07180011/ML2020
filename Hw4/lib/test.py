import numpy as np
from lib.preprocess import embedding_x, load_testing_data

def test(testing_path='./data/test_x.npy', output_path='./predict.csv', model_path='./model/model.h5'):


    from keras.models import load_model

    test_x = load_testing_data(testing_path)
    with open('./data/word2idx.txt', 'r') as fp:
        buf = fp.read()
        fp.close()
    word2idx = eval(buf)
    test_x = np.array(embedding_x(word2idx, test_x))

    prediction = load_model(model_path).predict(test_x)
    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i, y in  enumerate(prediction):
            f.write('{},{}\n'.format(i, 0 if y<=0.5 else 1))
