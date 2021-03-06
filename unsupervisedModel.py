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
from tensorflow_core.compiler.tf2xla.python.xla import lt


class unsupervisedModel(object):

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
        dfs = [dataset_5M, dataset_P99, dataset_P95, dataset_failedAction,dataset_SuccessAction]
        dataset = reduce(lambda left, right: pd.merge(left, right, on='date'), dfs)
        dataset.drop_duplicates(subset=None, inplace=True)

        self.dates = pd.to_datetime(dataset['date'], format='%Y-%m-%dT%H:%M:%S')
        dataset.drop(['date'], 1, inplace=True)
        dataset.set_index(self.dates, inplace=True)
        self.dataset = dataset
        return dataset



        # normalize features
    def normalize_features(self, df):
        for feature in df:
            # ensure all data is float
            df[feature] = df[feature].astype('float32')
            # normalize features
            # scaler = StandardScaler()
            scaler = MinMaxScaler(feature_range=(0, 1))
            df[feature] = scaler.fit_transform(df[[feature]])
            self.scaler = scaler
        return df

    def make_time_steps_data(self, values, n_time_steps):
        # split into input and outputs
        values_X = values[:len(values)-n_time_steps, :]
        values_y = values[n_time_steps:, :]
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
        train_y = train_y.reshape((train_y.shape[0], 1, train_y.shape[1]))
        test_y = test_y.reshape((test_y.shape[0], 1, test_y.shape[1]))

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

    def create_model(self, epochs_in, bach_size_in):
        # define model
        self.model = Sequential()
        self.model.add(LSTM(args.n_nodes, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
        self.model.add(Dropout(rate=0.2))
        self.model.add(RepeatVector(n=self.train_X.shape[1]))
        self.model.add(LSTM(units=64, return_sequences=True))
        self.model.add(Dropout(rate=0.2))
        self.model.add(TimeDistributed(Dense(self.train_X.shape[2])))
        self.model.compile(optimizer='adam', loss='mae')
        # fit network
        self.history = self.model.fit(self.train_X, self.train_y, epochs=epochs_in, batch_size=bach_size_in, validation_split=0.1, shuffle=False)

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

    def prediction(self, test):
        # X_train_pred = self.model.predict(self.train_X)
        # train_mae_loss = np.mean(np.abs(X_train_pred - self.train_X), axis=1)
        # train_mae_loss_avg_vector = np.mean(train_mae_loss, axis=1)
        # sns.distplot(train_mae_loss_avg_vector, bins=50, kde=True)
        # plt.show()

        X_test_pred = self.model.predict(self.test_X)

        test = test[:len(X_test_pred)]

        test_mae_loss = np.mean(np.abs(X_test_pred - self.test_y), axis=1) #test_y or test_X????
        test_mae_loss_avg_vector = np.mean(test_mae_loss, axis=1)
        test_score_df = pd.DataFrame(index=test.index)
        # test_score_df = pd.DataFrame()
        test_score_df['loss'] = test_mae_loss_avg_vector
        THRESHOLD = np.mean(test_mae_loss_avg_vector) + 3*np.std(test_mae_loss_avg_vector)
        exp_mean = test_score_df['loss'].ewm(com=0.5).mean() + 1 * np.std(test_mae_loss_avg_vector)
        rolling_mean = test_score_df['loss'].rolling(window=120).mean() + 3*np.std(test_mae_loss_avg_vector)
        test_score_df['rolling_mean'] = rolling_mean
        test_score_df['exp_mean'] = exp_mean
        test_score_df['threshold'] = THRESHOLD
        test_score_df['global_anomaly'] = test_score_df.loss > test_score_df.exp_mean
        # test_score_df['success_action'] = test['success_action']

        self.test_score_df = test_score_df
        self.test = test
        self.test_mae_loss = test_mae_loss
        # plot
        plt.plot(test_score_df.index, test_score_df.loss, label='avg loss')
        plt.plot(test_score_df.index, test_score_df.rolling_mean, label='rolling_mean')
        plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
        plt.plot(test_score_df.index, test_score_df.exp_mean, label='exp_mean')
        plt.xticks(rotation=25)
        plt.legend()
        plt.show()

    def anomalies(self, metrics_names):
        test_score_df = self.test_score_df
        metric_index = 0
        for metric in metrics_names:
            test_score_df[metric+'_loss'] = self.test_mae_loss[:, metric_index]
            metric_index = metric_index + 1
            THRESHOLD = np.mean(test_score_df[metric+'_loss']) + 3 * np.std(test_score_df[metric+'_loss'])
            exp_mean = test_score_df[metric+'_loss'].ewm(com=0.5).mean() + 1 * np.std(test_score_df[metric+'_loss'])
            # rolling_mean = test_score_df[metric+'_loss'].rolling(window=120).mean() + 3 * np.std(test_score_df[metric+'_loss'])
            test_score_df['self_anomaly'] = test_score_df[metric+'_loss'] > exp_mean
            test_score_df[metric] = self.test[metric]
            global_anomalies = self.test_score_df[self.test_score_df.global_anomaly == True]
            global_anomalies.head()
            self_anomalies = self.test_score_df[self.test_score_df.self_anomaly == True]
            self_anomalies.head()
            both_anomalies = self.test_score_df[self.test_score_df.self_anomaly & self.test_score_df.global_anomaly]
            both_anomalies.head()

            test_p = self.test[1200:]
            self_anomalies_p = self_anomalies[1200:]
            global_anomalies_p = global_anomalies[1200:]
            both_anomalies_p = both_anomalies[1200:]


            plt.plot(
                self.test.index,
                # test_p.index,
                # self.scaler.inverse_transform(self.test.success_action),

                # self.test.success_action,
                # label='success_action'
                # test_p[metric],
                self.test[metric],
                label=metric
            );


            sns.scatterplot(
                self_anomalies.index,
                # self_anomalies_p.index,
                # self.scaler.inverse_transform(anomalies.success_action),

                # global_anomalies.success_action,
                # self_anomalies_p[metric],
                self_anomalies[metric],
                color=sns.color_palette()[2],
                s=52,
                label='local_anomaly'
            )
            sns.scatterplot(
                global_anomalies.index,
                # global_anomalies_p.index,
                # self.scaler.inverse_transform(anomalies.success_action),

                # global_anomalies.success_action,
                # global_anomalies_p[metric],
                global_anomalies[metric],
                color=sns.color_palette()[8],
                s=52,
                label='global_anomaly'
            )
            sns.scatterplot(
                both_anomalies.index,
                # both_anomalies_p.index,
                # self.scaler.inverse_transform(anomalies.success_action),

                # global_anomalies.success_action,
                # both_anomalies_p[metric],
                both_anomalies[metric],
                color=sns.color_palette("husl", 8)[7],
                s=52,
                label='both_anomaly'
            )
            plt.xticks(rotation=25)
            plt.legend()
            plt.show()

    def plots(self, metrics_names, start_date, end_date):
        test_score_df = self.test_score_df

        metric_index = 0
        for metric in metrics_names:
            self.test_score_df[metric + '_loss'] = self.test_mae_loss[:, metric_index]
            metric_index = metric_index + 1

        test_score_df = self.test_score_df.loc[start_date:end_date]
        self.test = self.test[start_date:end_date]

        # metric_index = 0
        for metric in metrics_names:
            # test_score_df[metric + '_loss'] = self.test_mae_loss[:, metric_index]
            THRESHOLD = np.mean(test_score_df[metric + '_loss']) + 3 * np.std(test_score_df[metric + '_loss'])
            exp_mean = test_score_df[metric + '_loss'].ewm(com=0.5).mean() + 1 * np.std(test_score_df[metric + '_loss'])
            # rolling_mean = test_score_df[metric+'_loss'].rolling(window=120).mean() + 3 * np.std(test_score_df[metric+'_loss'])
            self.test_score_df['self_anomaly'] = test_score_df[metric + '_loss'] > exp_mean
            self.test_score_df[metric] = self.test[metric]
            global_anomalies = self.test_score_df[self.test_score_df.global_anomaly == True]
            global_anomalies.head()
            self_anomalies = self.test_score_df[self.test_score_df.self_anomaly == True]
            self_anomalies.head()
            both_anomalies = self.test_score_df[self.test_score_df.self_anomaly & self.test_score_df.global_anomaly]
            both_anomalies.head()

            test_p = self.test[1200:]
            self_anomalies_p = self_anomalies[1200:]
            global_anomalies_p = global_anomalies[1200:]
            both_anomalies_p = both_anomalies[1200:]

            self.self_anomalies = self_anomalies
            self.global_anomalies = global_anomalies
            self.both_anomalies = both_anomalies

            plt.plot(
                self.test.index,
                # test_p.index,
                # self.scaler.inverse_transform(self.test.success_action),

                # self.test.success_action,
                # label='success_action'
                # test_p[metric],
                self.test[metric],
                label=metric
            );

            sns.scatterplot(
                self_anomalies.index,
                # self_anomalies_p.index,
                # self.scaler.inverse_transform(anomalies.success_action),

                # global_anomalies.success_action,
                # self_anomalies_p[metric],
                self_anomalies[metric],
                color=sns.color_palette()[2],
                s=52,
                label='local_anomaly'
            )
            sns.scatterplot(
                global_anomalies.index,
                # global_anomalies_p.index,
                # self.scaler.inverse_transform(anomalies.success_action),

                # global_anomalies.success_action,
                # global_anomalies_p[metric],
                global_anomalies[metric],
                color=sns.color_palette()[8],
                s=52,
                label='global_anomaly'
            )
            sns.scatterplot(
                both_anomalies.index,
                # both_anomalies_p.index,
                # self.scaler.inverse_transform(anomalies.success_action),

                # global_anomalies.success_action,
                # both_anomalies_p[metric],
                both_anomalies[metric],
                color=sns.color_palette("husl", 8)[7],
                s=52,
                label='both_anomaly'
            )
            plt.xticks(rotation=25)
            plt.legend()
            plt.show()

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

    plt.legend(["5M", "P99", "P95", "failed action", "success action"])
    plt.xticks(rotation='vertical')

    plt.show()

    TIMESTESPS = 6

    df = UM.normalize_features(df)
    values = df.values

    train_size = int(args.train_size * (len(df)))
    test_size = len(df) - train_size
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
    test_x = test[0:len(test)-TIMESTESPS]

    UM.split_train_test(values, args.train_size, time_steps=TIMESTESPS)
    UM.create_model(args.epochs, args.batch_size)
    UM.plot_history()
    UM.prediction(test)
    metrics_names = df.columns

    UM.anomalies(metrics_names)
    
    start_date = pd.datetime(2020, 6, 1)
    end_date = pd.datetime(2020, 6, 2)
    UM.plots(metrics_names, start_date, end_date)



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
