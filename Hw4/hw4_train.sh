mkdir -p model/
wget https://www.csie.ntu.edu.tw/~b06902017/hw4/embedding_matrix.npy
mv embedding_matrix.npy data/
python3 main.py -w preprocess -f $1 $2
