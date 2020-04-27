import os
import numpy as np
from argparse import ArgumentParser

if __name__ == '__main__':

    np.random.seed(880301)

    parser = ArgumentParser()
    parser.add_argument('-w', '--work', help='work to execute', dest='work', default=None)
    parser.add_argument('-f', '--file', help='process file', nargs='+', dest='file')
    parser.add_argument('-e', '--epochs', help='number of training epochs', dest='epochs', type=int, default=5)
    parser.add_argument('-b', '--batch_size', help='batch size of training', dest='batch_size', type=int, default=256)
    parser.add_argument('-m', '--model', help='model path', dest='model', default='./model/model.h5')
    parser.add_argument('-p', '--plot', help='plot accuracy and loss', dest='plot', type=bool, default=False)
    parser.add_argument('-s', '--seed', help='random seed', dest='random_seed', type=int, default=880301)
    args = parser.parse_args()

    if args.work == None:
        print('must specify work')

    elif args.work == 'preprocess' or args.work =='pre':
        from lib.preprocess import do_preprocess
        if args.file != None:
            do_preprocess(args.file[0], args.file[1])
        else:
            do_preprocess(test_exist=True)
        os.system('python3 main.py -w train -e 5 -b 256')

    elif args.work == 'test':
        from lib.test import test
        test(args.file[0], args.file[1])

    elif args.work == 'train':
        from lib.train import train
        history = train(model_path=args.model, epochs=args.epochs, batch_size=args.batch_size, seed=args.random_seed)
        if args.plot == True:
            from lib.plot import plot
            plot(history)

    elif args.work == 'semi':
        from lib.semiParse import semiParse
        semiParse()
