from functools import reduce

import numpy as np
import tensorflow as tf
import os
from os import path

from keras import Sequential, Input, Model
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from pasta.augment import inline
from tensorflow import keras, config
import pandas as pd
import seaborn as sns
from pylab import rcParams, matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc, pyplot
from pandas.plotting import register_matplotlib_converters
import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import LSTM_Model


class unsupervisedModel(object):

    def import_data(self, dataPath):
        # combine all dates in 5M
        f_path = dataPath + os.sep + "recommendation_requests_5m_rate_dc"
        dfs = [pd.read_csv(path.join(f_path, x)) for x in os.listdir(f_path) if path.isfile(path.join(f_path, x))]
        dataset_5M = pd.concat(dfs)
        dataset_5M.columns = ['date', 'feature1']
        # combine all dates in P99
        f_path = dataPath + os.sep + "trc_requests_timer_p99_weighted_dc"
        dfs = [pd.read_csv(path.join(f_path, x)) for x in os.listdir(f_path) if path.isfile(path.join(f_path, x))]
        dataset_P99 = pd.concat(dfs)
        dataset_P99.columns = ['date', 'feature2']
        # combine all dates in P95
        f_path = dataPath + os.sep + "trc_requests_timer_p95_weighted_dc"
        dfs = [pd.read_csv(path.join(f_path, x)) for x in os.listdir(f_path) if path.isfile(path.join(f_path, x))]
        dataset_P95 = pd.concat(dfs)
        dataset_P95.columns = ['date', 'feature3']
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
        dfs = [dataset_5M, dataset_P99, dataset_P95, dataset_SuccessAction]
        dataset = reduce(lambda left, right: pd.merge(left, right, on='date'), dfs)
        dataset.drop_duplicates(subset=None, inplace=True)
        dataset.drop('date', 1)
        dataset.drop(dataset.columns[[0]], axis=1, inplace=True)

        self.dataset = dataset
        return dataset



        # normalize features
    def normalize_features(self, df):
        for feature in df:
            # ensure all data is float
            df[feature] = df[feature].astype('float32')
            # normalize features
            scaler = StandardScaler()
            scaler = scaler.fit(df[[feature]])
            df[feature] = scaler.transform(df[[feature]])
        return df

    def make_time_steps_data(self, values, n_time_steps):
        # split into input and outputs
        values_X = values[:len(values)-n_time_steps, :-1]
        values_y = values[n_time_steps:, -1]
        return values_X, values_y

    def split_train_test(self, values, train_size, time_steps):
        n_time_steps = time_steps
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

    def create_dataset(self, X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)

    def create_model(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y
        # define model
        self.model = Sequential()
        self.model.add(LSTM(args.n_nodes, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
        self.model.add(Dropout(rate=0.2))
        self.model.add(RepeatVector(n=train_X.shape[1]))
        self.model.add(LSTM(units=64, return_sequences=True))
        self.model.add(Dropout(rate=0.2))
        self.model.add(TimeDistributed(Dense(self.train_X.shape[2])))
        self.model.compile(optimizer='adam', loss='mse')
        # fit network
        history = self.model.fit(self.train_X, self.train_y, epochs=10, batch_size=32, validation_split=0.1,shuffle=False)

    # returns train, inference_encoder and inference_decoder models
    def define_models(self, n_input, n_output, n_units):
        # define training encoder
        encoder_inputs = Input(shape=(None, n_input))
        encoder = LSTM(n_units, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
        # define training decoder
        decoder_inputs = Input(shape=(None, n_output))
        decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(n_output, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        # define inference encoder
        encoder_model = Model(encoder_inputs, encoder_states)
        # define inference decoder
        decoder_state_input_h = Input(shape=(n_units,))
        decoder_state_input_c = Input(shape=(n_units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        # return all models
        return model, encoder_model, decoder_model

    def plot_history(self):
        # plot history
        pyplot.plot(self.history.history['loss'], label='train')
        pyplot.plot(self.history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

def main(args=None):

    UM = unsupervisedModel()
    dataPath = args.path

    register_matplotlib_converters()
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    rcParams['figure.figsize'] = 22, 10
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    df = UM.import_data(dataPath)

    plt.plot(df)

    plt.legend(["5M", "P99", "P95", "success action"])
    plt.xticks(rotation='vertical')

    plt.show()

    values = df.values

    train_size = int(len(df) * 0.95)
    test_size = len(df) - train_size
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
    print(train.shape, test.shape)

    df = UM.normalize_features(df)
    TIME_STEPS = 1

    # reshape to [samples, time_steps, n_features]
    x = train.feature1

    train_X, train_y = UM.create_dataset(train[['feature1']], train.feature1, TIME_STEPS)
    X_test, y_test = UM.create_dataset(test[['feature1']], test.feature1, TIME_STEPS)

    print(train_X.shape)
    # UM.split_train_test(values, args.train_size, time_steps=1)
    UM.create_model(train_X, train_y)
    UM.plot_history()




if (__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Data path")
    parser.add_argument("train_size", type=float, help="Train size")
    # parser.add_argument("test_num", type=str, help="Test num")
    parser.add_argument("epochs", type=int, help="Epochs")
    parser.add_argument("batch_size", type=int, help="Batch Size")
    parser.add_argument("n_nodes", type=int, help="Nodes size")
    # parser.add_argument("initialize_size", type=int, help="Initialize size (days)")
    parser.add_argument("prediction_size", type=int, help="Prediction size (days)")
    # parser.add_argument("chunk_size", type=int, help="Chunk size (days)")
    # parser.add_argument("time_steps", type=int, help="Time steps output data")
    args = parser.parse_args()
    main(args)
