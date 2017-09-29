import h5py
import numpy as np
import pickle

# refs#: str
# data: 1x487--240x6x173
# datetime: 1x487--1x240
# keys: 1x6
# symbol: 1x487
filepath = '../data/data_800_1minute.mat'
data_table = {}

with h5py.File(filepath) as f:
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
                # symbols_one_day = f[entry_ref].value[0]
                # for symbol_ref in symbols_one_day:
                #     symbol_chars = f[symbol_ref].value
                #     symbol_str = ''.join([chr(x) for x in symbol_chars[:, 0]])
                values.append([''.join([chr(x) for x in f[symbol_ref].value[:, 0]])
                               for symbol_ref in f[entry_ref].value[0]])
                # values.append([f[symbol_ref].value for symbol_ref in symbols_one_day])
            data_table[k] = values
        # if k in data_table.keys() and isinstance(data_table[k][0], np.ndarray):
        #     print("%s : %s x %s" % (k, len(data_table[k]), data_table[k][0].shape))
    print('load data completed.')
output = open('../data/data_raw.pkl', 'wb')
pickle.dump(data_table, output)
output.close()
# tmpstr= ''.join([chr(x) for x in value1[:,0]])
