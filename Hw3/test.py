import os, sys, cv2
import numpy as np

def readfile(path):
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(128, 128))
    return x

np.random.seed(900104)
workspace_dir = sys.argv[1]
test_x = readfile(os.path.join(workspace_dir, "testing")).astype(float) / 255.0

from keras.models import load_model

prediction = load_model('model.h5').predict(test_x)
with open(sys.argv[2], 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, np.argmax(np.array(y))))
