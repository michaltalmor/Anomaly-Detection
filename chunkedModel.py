import os
from os import path

import joblib
import pandas as pd
from functools import reduce
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot
import matplotlib.pyplot as plt
import argparse
import csv


from tensorflow_core import metrics


class BatchModel(object):
    def __init__(self, test_num, dataPath):
        self.test_num = test_num
        self.dataPath = dataPath



    def import_data(self, dataPath):
        self.dataPath = dataPath
        # combine all dates in 5M
        f_path = dataPath + "\\recommendation_requests_5m_rate_dc"
        dfs = [pd.read_csv(path.join(f_path, x)) for x in os.listdir(f_path) if path.isfile(path.join(f_path, x))]
        dataset_5M = pd.concat(dfs)
        dataset_5M.columns = ['date', 'feature1']
        # combine all dates in P99
        f_path = dataPath + "\\trc_requests_timer_p99_weighted_dc"
        dfs = [pd.read_csv(path.join(f_path, x)) for x in os.listdir(f_path) if path.isfile(path.join(f_path, x))]
        dataset_P99 = pd.concat(dfs)
        dataset_P99.columns = ['date', 'feature2']
        # combine all dates in P95
        f_path = dataPath + "\\trc_requests_timer_p95_weighted_dc"
        dfs = [pd.read_csv(path.join(f_path, x)) for x in os.listdir(f_path) if path.isfile(path.join(f_path, x))]
        dataset_P95 = pd.concat(dfs)
        dataset_P95.columns = ['date', 'feature3']
        # combine all dates in failed_action
        f_path = dataPath + "\\total_failed_action_conversions"
        dfs = [pd.read_csv(path.join(f_path, x)) for x in os.listdir(f_path) if path.isfile(path.join(f_path, x))]
        dataset_failedAction = pd.concat(dfs)
        dataset_failedAction.columns = ['date', 'failed_action']
        # combine all dates in success_action
        f_path = dataPath + "\\total_success_action_conversions"
        dfs = [pd.read_csv(path.join(f_path, x)) for x in os.listdir(f_path) if path.isfile(path.join(f_path, x))]
        dataset_SuccessAction = pd.concat(dfs)
        dataset_SuccessAction.columns = ['date', 'success_action']
        # merge
        dfs = [dataset_5M, dataset_P99, dataset_P95, dataset_SuccessAction]
        dataset = reduce(lambda left, right: pd.merge(left, right, on='date'), dfs)
        dataset.drop_duplicates(subset=None, inplace=True)
        dataset.drop('date', 1)
        dataset.drop(dataset.columns[[0]], axis=1, inplace=True)
        self.dataset = dataset
        values = dataset.values

        return values

    # convert series to supervised learning
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    # normalize features
    def normalize_features(self, values):
        # ensure all data is float
        values = values.astype('float32')
        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        # frame as supervised learning
        reframed = self.series_to_supervised(scaled, 1, 1)
        values = reframed.values
        return values

    def split_train_test(self, values, trainSize, i):
        n_train_hours = int(len(self.dataset) * trainSize)
        train = values[:n_train_hours, :]
        test = values[n_train_hours:, :]
        # split into input and outputs
        chunk_size = int(len(self.dataset)/31)
        print(chunk_size)

        train_X, train_y = train[i*chunk_size:((i*chunk_size)+chunk_size-1), :-1], \
                           train[i*chunk_size:((i*chunk_size+chunk_size)-1), -1]
        test_X, test_y = test[:, :-1], test[:, -1]
        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

    def create_model(self):
        # design network
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
        self.model.add(Dense(1))
        self.model.compile(loss='mse', optimizer='adam', metrics=[metrics.mae])
        # fit network
        #self.history = self.model.fit(self.train_X, self.train_y, epochs=1000, batch_size=72,
         #                             validation_data=(self.test_X, self.test_y), verbose=2, shuffle=False)

    def fit_model(self):
        # fit network
        self.history = self.model.fit(self.train_X, self.train_y, epochs=50, batch_size=72,
                                      validation_data=(self.test_X, self.test_y), verbose=2, shuffle=False)
    def plot_history(self):
        # plot history
        pyplot.plot(self.history.history['loss'], label='train')
        pyplot.plot(self.history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()
        #pyplot.savefig(self.dataPath+'\\Plots'+self.test_num+'\\Loss.png')


    def make_a_prediction(self):
        # Predict
        Predict = self.model.predict(self.test_X, verbose=1)
        print(Predict)
        # Plot
        fig = plt.figure(2)
        plt.scatter(self.test_y, Predict)
        plt.show(block=False)
        fig.savefig(self.dataPath + '\\plot2.png')
        fig =plt.figure(3)
        Test, = plt.plot(self.test_y)
        Predict, = plt.plot(Predict)
        plt.legend([Predict, Test], ["Predicted Data", "Real Data"])
        plt.show()
        fig.savefig(self.dataPath + '\\plot3.png')

    def write_to_csv(self, path):
        with open(path+'results.csv', 'a') as results_file:
            employee_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            employee_writer.writerow(['John Smith', 'Accounting', 'November'])
            employee_writer.writerow(['Erica Meyers', 'IT', 'March'])

    def save_model(self):
        filename = 'finalized_model'+self.test_num+'.sav'
        joblib.dump(self.model, filename)


def main(args=None):
    test_num = args.test_num
    dataPath = args.path
    BM = BatchModel(test_num, dataPath)
    # dataPath = "D:\\לימודים\\שנה ג\\סמסטר ב\\התמחות\\kobiBryent_US\\US-20200401T133445Z-001\\US"
    values = BM.import_data(dataPath)

    if not os.path.isdir(dataPath + '\\' + test_num):
        os.mkdir(dataPath + '\\' + test_num)

    BM.dataPath = dataPath + '\\' + test_num
    values = BM.normalize_features(values)
    #chunks
    for i in range(21):
        print (i)

        BM.split_train_test(values, args.train_size,i)
        if i == 0:
            BM.create_model()
        BM.fit_model()

    BM.plot_history()
    BM.make_a_prediction()
    BM.save_model()




if (__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Data path")
    parser.add_argument("train_size", type=float, help="Train size")
    parser.add_argument("test_num", type=str, help="Test number")
    args = parser.parse_args()
    main(args)
