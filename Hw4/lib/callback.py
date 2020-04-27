import prettytable as pt

def make_table(epoch, epochs, train, val):
    with open('./data/progress.txt', 'w') as fp:
        tb = pt.PrettyTable()
        tb.field_names = ['Epoch', 'Train_Acc', 'Val_Acc']
        for i in range(max(0, len(train)-20), len(train), 1):
            tb.add_row([str(i+1) + '/' + str(epochs), str(train[i]), str(val[i])])
        fp.write(str(tb))
        fp.close()
    with open('./data/tmp_accuracy.txt', 'w') as fp:
        fp.write(','.join(str(s) for s in train) + ';' + ','.join(str(s) for s in val))
        fp.close()
