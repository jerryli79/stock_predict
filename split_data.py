import pickle
import random
import numpy as np
import os
import re


def load_data(filepath):
    pkl_file = open(filepath + 'data_raw.pkl', 'rb')
    data_dict = pickle.load(pkl_file)
    stock_data = data_dict['data']
    print("stock data : %s x %s" % (len(stock_data), stock_data[0].shape))
    symbol_data = data_dict['symbol']
    print("symbol data: %s x %s" % (len(symbol_data), len(symbol_data[0])))
    return stock_data, symbol_data


def data_formatted(filepath, data, period=1, save_flag=False, save_num=100):

    if period == 1:
        for i in range(len(data)):
            data[i] = np.transpose(data[i], (2, 0, 1))
        return data

    X_set = None
    # data: len=days
    formatted_data = []
    for i in range(len(data)):
        print(i)
        # data_transposed: 713x240x6
        data_1day = np.transpose(data[i], (2, 0, 1))
        X_1day = None
        for j in range(data_1day.shape[0]):
            # data_1day_1stock: 240x6
            data_1day_1stock = data_1day[j]
            sliding_window = 0
            X_1day_1stock = None
            while sliding_window != data_1day_1stock.shape[0]:
                # x: period x 6
                x = data_1day_1stock[sliding_window: sliding_window + period]
                x_open = x[0][0]
                x_close = x[-1][1]
                x_high = np.amax(x[:, 2])
                x_low = np.amin(x[:, 3])
                x_turnover = np.sum(x[:, 4])
                x_volume = np.sum(x[:, -1])
                x_new_period = np.asarray([[x_open, x_close, x_high, x_low, x_turnover, x_volume]])
                if X_1day_1stock is not None:
                    X_1day_1stock = np.concatenate((X_1day_1stock, x_new_period))
                else:
                    X_1day_1stock = x_new_period
                sliding_window = sliding_window + period
            if X_1day is not None:
                X_1day = np.concatenate((X_1day, [X_1day_1stock]))
            else:
                X_1day = [X_1day_1stock]
        formatted_data.append(X_1day)

        # TODO: the last file name has a bug.
        if save_flag is True and ((i + 1) % save_num == 0 or i == len(data) - 1):
            print('data %d to %d is saved.' % (i + 2 - save_num, i + 1))
            filename = filepath + 'formatted_data_' \
                       + str(period) + 'm_' \
                       + str(i + 2 - save_num) + 'd_' + str(i + 1) + 'd'
            np.save(filename, formatted_data)
            formatted_data = []


def split_data(filepath, training_num, sliding_window_size=2, step=1):

    # random.seed(10)
    days_for_training, days_for_testing, days_for_validation = split_day(487, training_num)

    if not os.path.exists(filepath):
        print('dir not exist!')
        return
    else:
        testing_x = None
        testing_y = None
        list_files = os.listdir(filepath)
        for filename in list_files:
            if filename.startswith('formatted_data'):
                stock_data = np.load(filepath + filename)
                begin_num, end_num = parse_params(filename)
                # process training data
                data_for_training = [stock_data[int(str(day)[-2:])]
                                     for day in days_for_training if begin_num <= day <= end_num]
                if len(data_for_training) != 0:
                    training_x, training_y = slice_data(data_for_training, sliding_window_size, step)
                    print(training_x.shape, training_y.shape)
                    # save training data
                    fn_suffix = filename[14:]
                    fn_training_x = filepath + 'training_x' + fn_suffix
                    fn_training_y = filepath + 'training_y' + fn_suffix
                    np.save(fn_training_x, training_x)
                    np.save(fn_training_y, training_y)
                # process testing data
                data_for_testing = [stock_data[int(str(day)[-2:])]
                                    for day in days_for_testing if begin_num <= day <= end_num]
                if len(data_for_testing) != 0:
                    testing_x_part, testing_y_part = slice_data(data_for_testing, sliding_window_size, step)
                    print(testing_x_part.shape, testing_y_part.shape)
                    if testing_x is not None:
                        testing_x = np.concatenate((testing_x, testing_x_part))
                    else:
                        testing_x = testing_x_part
                    if testing_y is not None:
                        testing_y = np.concatenate((testing_y, testing_y_part))
                    else:
                        testing_y = testing_y_part
                    print(testing_x.shape, testing_y.shape)
        # save testing data
        np.save(filepath + 'testing_x', testing_x)
        np.save(filepath + 'testing_y', testing_y)


def parse_params(filename):
    m = re.findall(r'm_(\d+)d_(\d+)', filename)
    return int(m[0][0]) - 1, int(m[0][1]) - 1


def split_day(days_num, training_num, validation_num=0):
    days_for_training = np.random.choice(days_num, training_num, replace=False).tolist()
    testing_num = days_num - training_num - validation_num
    tmp = [day for day in range(days_num) if day not in days_for_training]
    days_for_testing = np.random.choice(tmp, testing_num, replace=False).tolist()
    days_for_validation = [day for day in tmp if day not in days_for_testing]
    print("days for training: %d" % len(days_for_training))
    print(days_for_training)
    print("days for testing: %d" % len(days_for_testing))
    print(days_for_testing)
    print("days for validation: %d" % len(days_for_validation))
    print(days_for_validation)
    return days_for_training, days_for_testing, days_for_validation


def slice_data(data, sliding_window_size=2, step=1):
    if len(data) != 0 and (data[0].shape[1] - sliding_window_size) % step != 0:
        print("error on slice data!")
        return
    data_x = None
    data_y = None
    for i in range(len(data)):
        print(i)
        data_1day = data[i]
        sliding_window_begin = 0
        sliding_window_end = sliding_window_size
        while sliding_window_end <= data_1day.shape[1]:
            sliced_data_x = data_1day[:, sliding_window_begin: sliding_window_end - 1]
            sliced_data_y = data_1day[:, sliding_window_end - 1: sliding_window_end]
            if data_x is not None:
                data_x = np.concatenate((data_x, sliced_data_x))
            else:
                data_x = sliced_data_x
            if data_y is not None:
                data_y = np.concatenate((data_y, sliced_data_y))
            else:
                data_y = sliced_data_y
            sliding_window_begin += step
            sliding_window_end += step
        # print(data_x.shape)
        # print(data_y.shape)
    return data_x, data_y


def compute_ground_truth(filepath):

    if not os.path.exists(filepath):
        print('dir not exist!')
        return
    else:
        # process training ground truth
        list_files = os.listdir(filepath)
        for filename in list_files:
            if filename.startswith('training_x'):
                training_x_part = np.load(filepath + filename)
                training_y_part = np.load(filepath + filename.replace('x', 'y'))
                # y[:, 0, 2]: raw_rqalpha_close at last unit
                # x[:, -1, 1]: raw_rqalpha_close at the unit before y
                training_ground_truth_part = np.greater(training_y_part[:, 0, 1], training_x_part[:, -1, 1])
                fn_suffix = filename[10:]
                print(training_ground_truth_part.shape)
                np.save(filepath + 'training_ground_truth' + fn_suffix, training_ground_truth_part)
        # process testing ground truth
        testing_x = np.load(filepath + 'testing_x.npy')
        testing_y = np.load(filepath + 'testing_y.npy')
        testing_ground_truth = np.greater(testing_y[:, 0, 1], testing_x[:, -1, 1])
        print(testing_ground_truth.shape)
        np.save(filepath + 'testing_ground_truth', testing_ground_truth)
    return


def show_data_shape(filepath):
    list_files = os.listdir(filepath)
    list_files.sort()
    for filename in list_files:
        if filename.startswith('training_x') or filename.startswith('training_ground') \
                or filename.startswith('testing_x') or filename.startswith('testing_ground'):
            training_x = np.load(filepath + filename)
            print("%s: %s" % (filename, training_x.shape))


def run():
    filepath = '../data/'
    # stock_data, symbol_data = load_data(filepath)
    # data_formatted(filepath, stock_data, 5, True)
    # split_data(filepath, 477, 6, 2)
    # compute_ground_truth(filepath)
    show_data_shape(filepath)
run()