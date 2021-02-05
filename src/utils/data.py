import numpy as np
import pickle


def dataset_training(mode, test_patient, all_filenames, part_num = 4, max_len = 899):
    # filenames = split_list(all_filenames[mode][test_patient], wanted_parts=PART_NUM)
    filenames = np.random.permutation(all_filenames[mode][test_patient])
    number_files = len(filenames) // part_num if mode == "train" else len(filenames)
    X_total = []
    y_total = []

    for filename in filenames[:number_files]:
        with open(filename, "rb") as pickle_file:
            data = pickle.load(pickle_file)
            x = np.array(data["X"])
            y = np.array(data["y"])
            y = np.expand_dims(y, -1)
            if x.shape[0] == max_len:
                X_total.append(x)
                y_total.append(y)
            elif x.shape[0] < max_len:
                diff = max_len - x.shape[0]
                x = np.pad(x,pad_width=[(0, diff), (0,0), (0,0)], constant_values=0)
                X_total.append(x)
                y = np.pad(y,pad_width=[(0, diff), (0,0)],constant_values=0)
                y_total.append(y)
            elif x.shape[0] > max_len:
                for i in range(x.shape[0]//max_len):
                    start = i*max_len
                    end = (i+1)* max_len
                    if np.sum(y[start:end]) == 0:
                        continue
                    X_total.append(x[start:end, :, :])
                    y_total.append(y[start:end, :])

    return np.asarray(X_total), np.asarray(y_total)
