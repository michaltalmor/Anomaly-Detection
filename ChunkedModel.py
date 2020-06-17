import csv
import os
from itertools import zip_longest
from os import path

import joblib
import pandas as pd
from functools import reduce
from numpy.ma import array
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot
import matplotlib.pyplot as plt
import argparse
from keras import backend
import seaborn as sns
from pandas import concat
import numpy
from tensorflow_core import metrics
import datetime


class ChunkedModel(object):
    def __init__(self, test_num, dataPath):
        self.test_num = test_num
        self.dataPath = dataPath

        # TODO use the names of the metrics from the folders names)

    def import_data(self, dataPath):
        # combine all dates in 5M
        f_path = dataPath + os.sep + "recommendation_requests_5m_rate_dc"
        dfs = [pd.read_csv(path.join(f_path, x)) for x in os.listdir(f_path) if path.isfile(path.join(f_path, x))]
        dataset_5M = pd.concat(dfs)
        dataset_5M.columns = ['date', '5m']
        # combine all dates in P99
        f_path = dataPath + os.sep + "trc_requests_timer_p99_weighted_dc"
        dfs = [pd.read_csv(path.join(f_path, x)) for x in os.listdir(f_path) if path.isfile(path.join(f_path, x))]
        dataset_P99 = pd.concat(dfs)
        dataset_P99.columns = ['date', 'p99']
        # combine all dates in P95
        f_path = dataPath + os.sep + "trc_requests_timer_p95_weighted_dc"
        dfs = [pd.read_csv(path.join(f_path, x)) for x in os.listdir(f_path) if path.isfile(path.join(f_path, x))]
        dataset_P95 = pd.concat(dfs)
        dataset_P95.columns = ['date', 'p95']
        # combine all dates in failed_action
        f_path = dataPath + os.sep + "total_failed_action_conversions"
        dfs = [pd.read_csv(path.join(f_path, x)) for x in os.listdir(f_path) if path.isfile(path.join(f_path, x))]
        dataset_failedAction = pd.concat(dfs)
        dataset_failedAction.columns = ['date', 'failed_action']
        # combine all dates in success_action
        f_path = dataPath + os.sep + "total_success_action_conversions"
        dfs = [pd.read_csv(path.join(f_path, x)) for x in os.listdir(f_path) if path.isfile(path.join(f_path, x))]
        dataset_SuccessAction = pd.concat(dfs)
        dataset_SuccessAction.columns = ['date', 'success_action']
        # merge
        dfs = [dataset_5M,dataset_P99, dataset_P95,dataset_failedAction, dataset_SuccessAction]
        dataset = reduce(lambda left, right: pd.merge(left, right, on='date'), dfs)
        dataset.drop_duplicates(subset=None, inplace=True)


        dataset = self.add_features(dataset)
        #dataset["time_steps_success_action"] = dataset["success_action"]

        self.dates = pd.to_datetime(dataset['date'], format='%Y-%m-%dT%H:%M:%S')
        dataset.drop('date', 1)
        dataset.drop(dataset.columns[[0]], axis=1, inplace=True)

        self.dataset = dataset
        values = dataset.values
        return values

    def add_features(self, dataset):
        dataset = self.add_is_rush_hour(dataset)
        dataset = self.add_isWeekend_feature(dataset)
        dataset = self.add_trend(dataset)
        #dataset = self.add_anomaly(dataset)
        dataset = self.add_multiply(dataset)
        dataset = self.add_time_feature(dataset)
        dataset = self.drop_low_corr_feature(dataset)
        #dataset = self.add_anomaly(dataset)
        list = dataset.columns.tolist()  # list the columns in the df
        list.insert(len(list), list.pop(list.index('success_action')))  # Assign new position (i.e. 8) for "success_action"
        dataset = dataset.reindex(columns=list)  # Now move 'success_action' to ist new position
        return dataset

    def add_anomaly(self, dataset):
        anomaly = dataset["success_action"]
        is_anomaly = [1 if num == 0 else 0 for num in anomaly]
        dataset["is_anomaly"] = is_anomaly
        return dataset

    def drop_low_corr_feature(self, dataset):
        prediction = dataset["success_action"][args.time_steps:]
        dataset_to_corr = dataset[:-args.time_steps]
        dataset_to_corr["prediction"] = prediction
        corr = dataset_to_corr.corr()["prediction"]
        corr = corr.abs()
        print(corr)

        for name in dataset_to_corr.columns:
            if (name != "time" and name != "date" and corr[name] < 0.8):
                dataset.drop(columns=[name], inplace=True)
        #dataset.drop(columns=["is_rush_hour * is_weekend"])
        return dataset

    def add_time_feature(self, dataset):
        # dataset['day'] = dataset['date'].str.split(' ', expand=True)[0]
        dataset['time'] = dataset['date'].str.split(' ', expand=True)[1]
        dataset['time'] = dataset['time'].str.replace(':', '')
        return dataset

    def add_isWeekend_feature(self, dataset):
        dataset['is_weekend'] = dataset['date'].str.split(' ', expand=True)[0]
        dataset['is_weekend'] = pd.to_datetime(dataset['is_weekend'], format='%Y-%m-%d')
        dataset['is_weekend'] = dataset['is_weekend'].dt.dayofweek
        is_weekend = dataset['is_weekend'].apply(lambda x: 1 if x >= 5.0 else 0)
        dataset['is_weekend'] = is_weekend
        return dataset

    def add_is_rush_hour(self, dataset):
        requests = dataset["5m"]
        threshold_value = requests.sort_values()[numpy.math.floor(0.8 * requests.size)]
        is_rush_hour = [1 if num > threshold_value else 0 for num in requests]
        dataset["is_rush_hour"] = is_rush_hour
        return dataset

    def add_trend(self, dataset):
        feature_names = ["5m", "p95", "p99"]
        i = 0
        for feature in feature_names:
            i += 1
            x = dataset[feature]
            trend = [b - a for a, b in zip(x[::1], x[1::1])]
            trend.append(0)
            dataset["trend_" + feature] = trend
        return dataset

    def add_multiply(self, dataset):
        feature_names1 = dataset.columns
        feature_names2 = dataset.columns

        for feature1 in feature_names1:
            feature_names2 = feature_names2[1:]
            for feature2 in feature_names2:
                if (feature1 != feature2 and feature1 !=  "date" and feature2 != "date"):
                    to_add = dataset[feature1] * dataset[feature2]
                    dataset[feature1 + " * " + feature2] = to_add
        return dataset

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
        return scaled

    def get_init_sequences(self, sequences, n_days, n_data_per_day):
        # find the end of this pattern
        end_i_data = n_days*n_data_per_day
        # gather input and output parts of the pattern
        seq_1, seq_2 = sequences[0:end_i_data, :], sequences[end_i_data:, :]
        return seq_1, seq_2

    def get_predict_sequences(self, sequences, n_days, n_data_per_day):
        # find the end of this pattern
        start_i_data = len(sequences)-(n_days*n_data_per_day)
        # gather input and output parts of the pattern
        seq_data, seq_data_to_predict = sequences[0:start_i_data, :], sequences[start_i_data:, :]
        self.pradict_data_dates = self.dates.values[start_i_data:]
        return seq_data, seq_data_to_predict

    def split_sequences(self, sequences, n_steps):
        X = list()
        for i in range(0, len(sequences), n_steps):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the dataset
            if end_ix > len(sequences):
                end_ix = len(sequences)
                # gather input and output parts of the pattern
                seq_x = sequences[i:end_ix, :]
                X.append(seq_x)
                break
            # gather input and output parts of the pattern
            seq_x = sequences[i:end_ix, :]
            X.append(seq_x)
        return array(X)

    def make_time_steps_data(self, values, n_time_steps):
        # split into input and outputs
        values_X = values[:len(values)-n_time_steps, :-1]
        values_y = values[n_time_steps:, -1]
        return values_X, values_y


    def split_train_test(self, values, train_size):
        n_time_steps = args.time_steps
        values_X, values_y = self.make_time_steps_data(values, n_time_steps)
        n_train_hours = int((len(values_X)) * train_size)
        train_X = values_X[:n_train_hours, :]
        train_y = values_y[:n_train_hours]

        test_X = values_X[n_train_hours:, :]
        test_y = values_y[n_train_hours:]

        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

    def rmse(selfs, y_true, y_pred):
        return backend.sqrt(backend.mean(backend.square(y_pred-y_true), axis=1))

    def create_model(self):
        # design network
        self.model = Sequential()
        self.model.add(LSTM(args.n_nodes, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
        self.model.add(Dense(1))
        self.model.compile(loss=self.rmse, optimizer='adam', metrics=[metrics.mae])


    def fit_model(self, epochs_in, batch_size_in):
        # fit network
        self.history = self.model.fit(self.train_X, self.train_y, epochs=epochs_in, batch_size=batch_size_in,
                                      validation_data=(self.test_X, self.test_y), verbose=2, shuffle=False)
    def plot_history(self):
        # plot history
        pyplot.plot(self.history.history['loss'], label='train')
        pyplot.plot(self.history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

    def make_a_prediction(self, values):
        test_X = values[:, :-1]
        test_y = values[:, -1]
        # reshape input to be 3D [samples, timesteps, features]
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        self.test_X = test_X
        self.test_y = test_y

        # Predict
        Predict = self.model.predict(self.test_X, verbose=1)
        #print(Predict)

        #csv results file
        with open('results.csv', 'w') as file:
            writer = csv.writer(file)
            d = [Predict,  (map(lambda x: [x], self.test_y))]
            export_data = zip_longest(*d, fillvalue='')
            writer.writerows(export_data)

        # Plot
        sns.set()
        fig, ax = plt.subplots(figsize=(11, 11))
        sns.heatmap(self.dataset.corr(), cmap='coolwarm')

        self.put_dates_in_plot(self.test_y, Predict)

        fig = plt.figure(4)
        #Test, = plt.plot(self.pradict_data_dates, self.test_y)
        Test, = plt.plot(self.test_y)
        #Predict, = plt.plot(self.pradict_data_dates, Predict)
        Predict, = plt.plot( Predict)
        plt.legend([Predict, Test], ["Predicted Data", "Real Data"])
        plt.xticks(rotation='vertical')

        plt.show()
        fig.savefig(self.dataPath + '\\plot4.png')

        #sns.regplot(Test, Predict)
        #sns.despine()

    def put_dates_in_plot(self, test_y, Predict):
        fig = plt.figure(4)
        predict_data_dates = self.pradict_data_dates[1:]
        test_y = self.test_y[1:]
        Predict = Predict[1:]
        Test, = plt.plot(predict_data_dates, test_y)
        Predict, = plt.plot(predict_data_dates, Predict)
        plt.legend([Predict, Test], ["Predicted Data", "Real Data"])
        plt.xticks(rotation='vertical')

        plt.show()


    def save_model(self):
        filename = 'finalized_model'+self.test_num+'.sav'
        joblib.dump(self.model, filename)


def main(args=None):
    test_num = args.test_num
    dataPath = args.path
    CM = ChunkedModel(test_num, dataPath)
    values = CM.import_data(dataPath)

    if not os.path.isdir(dataPath + '\\' + test_num):
        os.mkdir(dataPath + '\\' + test_num)

    CM.dataPath = dataPath + '\\' + test_num
    values = CM.normalize_features(values)


# chunks
    # chunks
    n_days = 37 # in data path
    n_data_per_day = int(len(values) / n_days)
# split the last days to predict
    seq_data, seq_data_to_predict = CM.get_predict_sequences(values, args.prediction_size, n_data_per_day)
# split the init days to create the model
    init_seq, addition_data_seq = CM.get_init_sequences(seq_data, args.initialize_size, n_data_per_day)
    CM.split_train_test(init_seq, args.train_size)
    CM.create_model()
    CM.fit_model(args.epochs, args.batch_size)
    chunk_size = args.chunk_size #n_days
# data array
    chunked_data = CM.split_sequences(addition_data_seq, chunk_size*n_data_per_day)

    for chunk in chunked_data:
        if (len(chunk)<args.time_steps):
            break
        CM.split_train_test(chunk, args.train_size)
        CM.fit_model(args.epochs, args.batch_size)

    CM.plot_history()
    CM.make_a_prediction(seq_data_to_predict)
    CM.save_model()




if (__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Data path")
    parser.add_argument("train_size", type=float, help="Train size")
    parser.add_argument("test_num", type=str, help="Test num")
    parser.add_argument("epochs", type=int, help="Epochs")
    parser.add_argument("batch_size", type=int, help="Batch Size")
    parser.add_argument("n_nodes", type=int, help="Nodes size")
    parser.add_argument("initialize_size", type=int, help="Initialize size (days)")
    parser.add_argument("prediction_size", type=int, help="Prediction size (days)")
    parser.add_argument("chunk_size", type=int, help="Chunk size (days)")
    parser.add_argument("time_steps", type=int, help="Time steps output data")
    args = parser.parse_args()
    main(args)
