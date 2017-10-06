import h5py
import pickle
import numpy as np


def reformat(filepath):
    # refs#: str
    # data: 1x487--240x6x173
    # datetime: 1x487--1x240
    # keys: 1x6
    # symbol: 1x487
    mat_file = filepath + 'data_800_1minute.mat'
    data_table = {}

    with h5py.File(mat_file) as f:
        for k, v in f.items():
            values = list()
            if k == 'data':
                dataset = v.value[0]
                for entry_ref in dataset:
                    data_one_day = f[entry_ref].value
                    values.append(data_one_day)
                data_table[k] = values
            if k == 'symbol':
                dataset = v.value[0]
                for entry_ref in dataset:
                    values.append([''.join([chr(x) for x in f[symbol_ref].value[:, 0]])
                                   for symbol_ref in f[entry_ref].value[0]])
                data_table[k] = values
        print('load data completed.')
    output = open('data/data_raw1.pkl', 'wb')
    pickle.dump(data_table, output)
    output.close()
    print('reformat done!')
    return data_table


def load(filepath):
    pkl_file = open(filepath + 'data_raw1.pkl', 'rb')
    data_dict = pickle.load(pkl_file)
    stock_data = data_dict['data']
    print("stock data : %s x %s" % (len(stock_data), stock_data[0].shape))
    stock_symbols = data_dict['symbol']
    print("symbol data: %s x %s" % (len(stock_symbols), len(stock_symbols[0])))
    return stock_data, stock_symbols


def reorganize(stock_data, stock_symbols):
    stock_symbols_unique = set()
    for stock_symbol_1day in stock_symbols:
        stock_symbols_unique.update(set(stock_symbol_1day))
    stock_symbols_unique = list(stock_symbols_unique)

    stock_symbols_num = len(stock_symbols_unique)
    days = len(stock_symbols)
    samples_1day = 240
    # 6 x (240 x 487) x 798
    data = np.zeros([stock_data[0].shape[1], samples_1day * days, stock_symbols_num])

    for day in range(days):
        stock_symbols_today = stock_symbols[day]
        invalid_symbols_today = list(set(stock_symbols_unique) - set(stock_symbols_today))
        data_idx = [stock_symbols_unique.index(stock_symbol) for stock_symbol in stock_symbols_today]
        nan_idx = [stock_symbols_unique.index(invalid_symbol) for invalid_symbol in invalid_symbols_today]
        data[:, day * samples_1day: (day + 1) * samples_1day, data_idx] = np.transpose(stock_data[day], (1, 0, 2))
        data[:, day * samples_1day: (day + 1) * samples_1day, nan_idx] = np.nan
    print('reorganize done!')
    return data, stock_symbols_unique


def normalize(data):
    for i in range(data.shape[0]):
        data[i, :, :] = z_score(data[i, :, :])
    print('normalization done!')
    return data


def z_score(sliced_data):
    mean = np.nanmean(sliced_data, axis=0)
    std = np.nanstd(sliced_data, axis=0)
    normalized_data = (sliced_data - mean) / std
    return normalized_data


def new_period(data, time_period=1):
    if time_period == 1:
        return data
    new_period_data = np.zeros([data.shape[0], int(data.shape[1] / time_period), data.shape[2]])
    # 23376 x 5
    slicer = np.reshape(np.arange(data.shape[1]), (int(data.shape[1] / time_period), time_period))
    # 4 x 23376 x 5 x 798
    sliced_data = data[[2, 3, 4, 5], :, :]
    sliced_data = sliced_data[:, slicer, :]
    # 23376 x 5 x 798 -> 23776 x 798
    # open, close, high, low, turnover, volume
    new_period_data[0, :, :] = data[0, 0::time_period, :]
    new_period_data[1, :, :] = data[1, time_period - 1::time_period, :]
    new_period_data[2, :, :] = np.nanmax(sliced_data[0, :, :], axis=1)
    new_period_data[3, :, :] = np.nanmin(sliced_data[1, :, :], axis=1)
    new_period_data[4, :, :] = np.nansum(sliced_data[2, :, :], axis=1)
    new_period_data[5, :, :] = np.nansum(sliced_data[3, :, :], axis=1)
    print('new_period done!')
    return new_period_data


def resample(data, sample_window, sample_step, filepath, stock_symbols_unique):
    filecount = 1
    resampled_stocks = dict()
    # 6 x 23376 x 798 --> 798 x 6 x 23376
    data = data.transpose([2, 0, 1])
    for i in range(data.shape[0]):
        # 6 x 23376
        data_1stock = data[i, :, :]
        mask = ~np.any(np.isnan(data_1stock), axis=0)
        data_1stock = data_1stock[:, mask]
        steps = int((data_1stock.shape[1] - sample_window) / sample_step)
        idx_matrix = np.arange(sample_window * steps).reshape([steps, sample_window])
        move_back_matrix = \
            np.arange(0, steps * (sample_window - sample_step), (sample_window - sample_step)).reshape([steps, 1])
        slicer = idx_matrix - move_back_matrix
        # 6 x steps x sample_step
        sliced_data = data_1stock[:, slicer]
        resampled_stocks[stock_symbols_unique[i]] = sliced_data
        if (i + 1) % 50 == 0 or i == data.shape[0] - 1:
            filename = filepath + "stock_" + str(filecount) + '.pkl'
            output = open(filename, 'wb')
            pickle.dump(resampled_stocks, output)
            output.close()
            print('part %d data saved!' % i)
            resampled_stocks = {}
            filecount += 1
        print('resampling %d done!' % i)


def run():
    filepath = 'data/'
    reformat(filepath)
    # 487 x (240 x 6 x 798), 485 x 798
    stock_data, stock_symbols = load(filepath)
    # 6 x (240 x 487) x 798
    reorganized_data, stock_symbols_unique = reorganize(stock_data, stock_symbols)
    # 6 x 23376 x 798
    new_period_data = new_period(reorganized_data, 5)
    # 6 x 23376 x 798
    normalized_data = normalize(new_period_data)
    # 798 x (6 x 11658 x 60)
    resample(normalized_data, 60, 2, filepath, stock_symbols_unique)

    print('All done!')

run()