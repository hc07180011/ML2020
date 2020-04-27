import numpy as np
import sys

test_data = np.genfromtxt(sys.argv[1], delimiter=',')
test_data = np.nan_to_num(test_data[:, 2:])

w = np.load("saved_w.npy")

ans = np.asarray(['id', 'value'])

for i in range(0, len(test_data), 18):
	data = []
	for j in range(18):
		for k in range(9):
			data.append(test_data[i+j][k])
	data.append(1)
	ans = np.vstack((ans, ['id_' + str(int(i/18)), np.dot(np.array(data), w)]))

np.savetxt(sys.argv[2], ans, fmt='%s', delimiter=',')