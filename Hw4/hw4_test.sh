mkdir -p model/
wget https://www.csie.ntu.edu.tw/~b06902017/hw4/embedding_matrix.npy
wget https://www.csie.ntu.edu.tw/~b06902017/hw4/model.h5
mv embedding_matrix.npy data/
mv model.h5 model/
python3 main.py -w test -f $1 $2
