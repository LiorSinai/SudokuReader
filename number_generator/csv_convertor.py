import pandas as pd
import numpy as np
import os
from PIL import Image


def convert_to_csv(input_dir, output_train_path, output_test_path, f_train=0.9):

    column_names = ["{:d}x{:d}".format(i, j) for i in range(1, 29) for j in range(1, 29)]
    column_names.insert(0, "label")

    rows = []
    for subdir in os.listdir(input_dir):
        current_path = os.path.join(input_dir, subdir)
        print("reading from folder " + current_path)
        input_paths = os.listdir(current_path)
        for fname in input_paths:
            image = Image.open(os.path.join(input_dir, subdir, fname))
            data_raw = np.asarray(image)
            data = data_raw.flatten()
            row = [int(subdir)]
            row.extend(data)
            rows.append(row)

    df = pd.DataFrame(rows, columns=column_names)
    # split train and test data
    idxs = df.index.values
    np.random.shuffle(idxs)
    n_train = int(f_train * df.shape[0])
    train_idxs = idxs[:n_train]
    test_idxs = idxs[n_train:]

    # write to csv
    df.loc[train_idxs, :].to_csv(output_train_path, index=False) #, mode='a', header=False)
    df.loc[test_idxs, :].to_csv(output_test_path, index=False) #, mode='a', header=False)
        
    print("done")




if __name__ == '__main__':
    input_dir = "..\\datasets\\74k_numbers_28x28"
    output_train_path = "..\\datasets\\74k_train.csv"
    output_test_path = "..\\datasets\\74k_test.csv"
    f_train = 0.9 

    convert_to_csv(input_dir, output_train_path, output_test_path, f_train=0.9)
    
    ## concatenation script
    # print("concatenating MNIST data")
    # df1 = pd.read_csv('../datasets/MNIST_data/mnist_train.csv')
    # df2 = pd.read_csv('../datasets/font_train.csv')
    # df = pd.concat([df1, df2])
    # df = df.sample(frac=1)
    # df.to_csv('../datasets/train_data.csv', index=False, mode='w')
    # print("done")