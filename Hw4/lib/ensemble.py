import numpy as np

with open('predict.csv') as f:
	next(f)
	y = np.array([line.strip('\n').split(',')[1] for line in f], dtype=int)
with open('predict1.csv') as f:
	next(f)
	y1 = np.array([line.strip('\n').split(',')[1] for line in f], dtype=int)
with open('predict2.csv') as f:
	next(f)
	y2 = np.array([line.strip('\n').split(',')[1] for line in f], dtype=int)
with open('predict3.csv') as f:
	next(f)
	y3 = np.array([line.strip('\n').split(',')[1] for line in f], dtype=int)
with open('predict4.csv') as f:
	next(f)
	y4 = np.array([line.strip('\n').split(',')[1] for line in f], dtype=int)
with open('predict5.csv') as f:
	next(f)
	y5 = np.array([line.strip('\n').split(',')[1] for line in f], dtype=int)

y = np.array([y, y1, y2, y3, y4, y5]).T

prediction = np.argmax(np.apply_along_axis(lambda x: np.bincount(x, minlength=2), axis=1, arr=y), axis=1)
with open('abc.csv', 'w') as f:
    f.write('id,label\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))
