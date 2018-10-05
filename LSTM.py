#!/usr/bin/python
# coding=utf-8
'''
every link a LSTM, then average
'''
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Input, Dropout, concatenate, subtract, add, Reshape, TimeDistributed, \
    RepeatVector
import cPickle
from collections import defaultdict
import numpy as np
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

gpu_id = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES')
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
KTF.set_session(session)


def predict(Y_test, Y_pred):
    mse = np.mean((Y_pred - Y_test) ** 2)
    mae = np.mean(np.abs(Y_pred - Y_test))
    # mape = np.mean(np.abs((Y_pred - Y_test)/Y_test)) * 100
    print("mse:{},\tmae:{}".format(mse, mae))
    return mse, mae


def load_data(
        window_size=12, resize_num=5, coarse_query_type_cnt=26):
    root_dir = "../pkl/"
    event_reshaped_traffic_list_file = root_dir +  "event_filtered_traffic_list_beijing_filt_2_normalized.pkl"
    input_event_file = root_dir + "event_beijing_final.txt"
    train_test_list_file = root_dir + "train_test_list_beijing.pkl"
    query_type_coarse_list_file = root_dir + "query_type_coarse_list_beijing.pkl"
    query_num_list_file = root_dir + "query_num_list_beijing_filt_2.pkl"
    query_time_list_file = root_dir + "query_time_list_beijing_filt_2.pkl"
    query_distribution_list_file = root_dir + "query_num_distribution_filt_2.pkl"
    # output
    traffic_data_one_lstm_file = root_dir + "traffic_data_residual_lstm_seq_beijing_filt_2.pkl"
    time_data_one_lstm_file = root_dir + "time_data_residual_lstm_seq_beijing.pkl"
    query_type_one_lstm_file = root_dir + "query_type_residual_lstm_seq_beijing.pkl"
    query_num_time_one_lstm_file = root_dir + "query_num_time_residual_lstm_seq_beijing_filt_2.pkl"
    query_distribution_one_lstm_file = root_dir + "query_distribution_residual_lstm_seq_beijing_filt_2.pkl"
    test_num_list_file = root_dir + "test_num_list_residual_lstm_seq_beijing_1205_filt_2.pkl"
    filtered_test_list_file = root_dir + "filtered_test_list_residual_lstm_seq_beijing_filt_2.pkl"

    if not os.path.exists(test_num_list_file):

        print "loading event ..."
        event_start_end = list()
        event_coord = list()
        event_cnt = 0
        for line in open(input_event_file):
            temp_arr = line.rstrip("\n").split("\t")
            event_start_end.append((int(temp_arr[0]), int(temp_arr[1])))
            event_coord.append((int(temp_arr[2]), int(temp_arr[3])))
            event_cnt += 1
        print "event_cnt is {}".format(event_cnt)

        print "loading train and test split ..."
        (train_list, test_list) = cPickle.load(open(train_test_list_file, "rb"))

        train_samples = 0
        test_samples = 0
        test_num_list = list()
        filtered_test_list = list()
        for temp_train_id in train_list:
            (temp_start, temp_end) = event_start_end[temp_train_id]
            temp_samples = temp_end - temp_start - 3 * window_size + 1
            if temp_samples < 1:
                print "temp_train_id is {}, temp_train_samples is {}".format(temp_train_id, temp_samples)
            else:
                train_samples += temp_samples
        print "train samples is {}".format(train_samples)
        for temp_test_id in test_list:
            (temp_start, temp_end) = event_start_end[temp_test_id]
            temp_samples = temp_end - temp_start - 3 * window_size + 1
            if temp_samples < 1:
                print "temp_test_id is {}, temp_test_samples is {}".format(temp_test_id, temp_samples)
            else:
                test_samples += temp_samples
                test_num_list.append(temp_samples)
                filtered_test_list.append(temp_test_id)
        print "test samples is {}".format(test_samples)

        print "loading query num data ..."
        query_num_list = cPickle.load(open(query_num_list_file, "rb"))
        max_query_num = 0
        min_query_num = 10000
        for temp_list in query_num_list:
            if max_query_num < np.max(temp_list):
                max_query_num = np.max(temp_list)
            if min_query_num > np.min(temp_list):
                min_query_num = np.min(temp_list)
        print "query_num_list max is {}, min is {}".format(max_query_num, min_query_num)

        print "loading event reshaped traffic ..."
        event_reshaped_traffic_list = cPickle.load(open(event_reshaped_traffic_list_file, "rb"))

        print "loading query type data ..."
        query_type_coarse_list = cPickle.load(open(query_type_coarse_list_file, "rb"))

        print "loading query time data ..."
        query_time_list = cPickle.load(open(query_time_list_file, "rb"))

        print "loading query distribution data ..."
        query_distribution_list = cPickle.load(open(query_distribution_list_file, "rb"))

        x_train = np.zeros((train_samples, 2 * window_size, resize_num), dtype=np.float)
        time_train = np.zeros((train_samples, 1), dtype=np.float)
        query_train = np.zeros((train_samples, coarse_query_type_cnt), dtype=np.float)
        query_num_time_train = np.zeros((train_samples, 2*window_size, 2), dtype=np.float)
        query_distribution_train = np.zeros((train_samples, 2*window_size, resize_num), dtype=np.float)
        y_train_all = np.zeros((train_samples, window_size, resize_num), dtype=np.float)
        x_test = np.zeros((test_samples, 2 * window_size, resize_num), dtype=np.float)
        time_test = np.zeros((test_samples, 1), dtype=np.float)
        query_test = np.zeros((test_samples, coarse_query_type_cnt), dtype=np.float)
        query_num_time_test = np.zeros((test_samples, 2*window_size, 2), dtype=np.float)
        query_distribution_test = np.zeros((test_samples, 2*window_size, resize_num), dtype=np.float)
        y_test_all = np.zeros((test_samples, window_size, resize_num), dtype=np.float)

        print "preparing training data ..."
        train_cnt = 0
        for temp_train_id in train_list:
            (temp_start, temp_end) = event_start_end[temp_train_id]
            # print "temp_train_id is {}, start is {}, end is {}".format(temp_train_id, temp_start, temp_end)
            temp_coarse_query_type_list = query_type_coarse_list[temp_train_id]
            temp_samples = temp_end - temp_start - 3 * window_size + 1
            if temp_samples < 1:
                continue
            temp_time_len = temp_end - temp_start + 1
            temp_time = np.zeros(temp_time_len, dtype=np.float)
            temp_query_num = query_num_list[temp_train_id]
            temp_query_time = query_time_list[temp_train_id]
            temp_query_distribution = query_distribution_list[temp_train_id]
            # print "temp_query_num len is {}".format(len(temp_query_num))
            for time_id in range(temp_time_len):
                temp_time[time_id] = float(time_id) / temp_time_len
            temp_traffic = event_reshaped_traffic_list[temp_train_id]
            # print "temp_samples is {}".format(temp_samples)
            for sample_id in range(temp_start, temp_start + temp_samples):
                x_train[train_cnt, :, :] = temp_traffic[sample_id: sample_id + 2 * window_size, :]
                time_train[train_cnt, 0] = temp_time[sample_id - temp_start]
                query_train[train_cnt, :] = temp_coarse_query_type_list[sample_id - temp_start + window_size]
                # print "train_id is {}, train_cnt is {}, temp_query_len is {}".format(temp_train_id, train_cnt, len(temp_query_num[(sample_id-temp_start+window_size): (sample_id-temp_start+2*window_size)]))
                query_num_time_train[train_cnt, :, 0] = temp_query_num[
                                                        sample_id - temp_start + window_size: sample_id - temp_start + 3 * window_size]
                query_num_time_train[train_cnt, :, 1] = temp_query_time[
                                                        sample_id - temp_start + window_size: sample_id - temp_start + 3 * window_size]
                query_distribution_train[train_cnt, :, :] = temp_query_distribution[
                                                            (sample_id - temp_start + window_size): (
                                                            sample_id - temp_start + 3 * window_size), :]
                y_train_all[train_cnt, :, :] = temp_traffic[sample_id + 2 * window_size: sample_id + 3 * window_size]
                train_cnt += 1

        print "preparing testing data ..."
        test_cnt = 0
        for temp_test_id in test_list:
            (temp_start, temp_end) = event_start_end[temp_test_id]
            temp_coarse_query_type_list = query_type_coarse_list[temp_test_id]
            temp_samples = temp_end - temp_start - 3 * window_size + 1
            if temp_samples < 1:
                continue
            temp_time_len = temp_end - temp_start + 1
            temp_time = np.zeros(temp_time_len, dtype=np.float)
            for time_id in range(temp_time_len):
                temp_time[time_id] = float(time_id) / temp_time_len
            temp_traffic = event_reshaped_traffic_list[temp_test_id]
            temp_query_num = query_num_list[temp_test_id]
            temp_query_time = query_time_list[temp_test_id]
            temp_query_distribution = query_distribution_list[temp_test_id]
            for sample_id in range(temp_start, temp_start + temp_samples):
                x_test[test_cnt, :, :] = temp_traffic[sample_id: sample_id + 2 * window_size, :]
                time_test[test_cnt, 0] = temp_time[sample_id - temp_start]
                query_test[test_cnt, :] = temp_coarse_query_type_list[sample_id - temp_start + window_size]
                query_num_time_test[test_cnt, :, 0] = temp_query_num[
                                                      sample_id - temp_start + window_size: sample_id - temp_start + 3 * window_size]
                query_num_time_test[test_cnt, :, 1] = temp_query_time[
                                                      sample_id - temp_start + window_size: sample_id - temp_start + 3 * window_size]
                query_distribution_train[test_cnt, :, :] = temp_query_distribution[
                                                           (sample_id - temp_start + window_size): (
                                                           sample_id - temp_start + 3 * window_size), :]
                y_test_all[test_cnt, :, :] = temp_traffic[sample_id + 2 * window_size: sample_id + 3 * window_size]
                test_cnt += 1

        print "normalizing the query data ..."
        max_type_num_train = np.max(query_train, axis=0)
        max_type_num_test = np.max(query_test, axis=0)
        for cnt in range(coarse_query_type_cnt):
            if max_type_num_train[cnt] > 0:
                query_train[:, cnt] /= max_type_num_train[cnt]
            if max_type_num_test[cnt] > 0:
                query_test[:, cnt] /= max_type_num_test[cnt]
        print "normalizing the query num time data ..."
        query_num_time_train[:, 0] /= np.max(query_num_time_train[:, 0])
        query_num_time_train[:, 1] /= np.max(query_num_time_train[:, 1])
        query_num_time_test[:, 0] /= np.max(query_num_time_test[:, 0])
        query_num_time_test[:, 1] /= np.max(query_num_time_test[:, 1])

        print "normalizing the query diatribution data ..."
        # print "query_distribution_train max is {}, min is {}".format(np.max(query_distribution_train), np.min(query_distribution_train))
        query_distribution_train /= np.max(query_num_time_train[:, 0])
        query_distribution_test /= np.max(query_num_time_train[:, 1])

        # saving data
        cPickle.dump((x_train, y_train_all, x_test, y_test_all), open(traffic_data_one_lstm_file, "wb"))
        cPickle.dump((query_train, query_test), open(query_type_one_lstm_file, "wb"))
        cPickle.dump((time_train, time_test), open(time_data_one_lstm_file, "wb"))
        cPickle.dump((query_num_time_train, query_num_time_test), open(query_num_time_one_lstm_file, "wb"))
        cPickle.dump((query_distribution_train, query_distribution_test), open(query_distribution_one_lstm_file, "wb"))
        cPickle.dump(test_num_list, open(test_num_list_file, "wb"))
        cPickle.dump(filtered_test_list, open(filtered_test_list_file, "wb"))
        return x_train, time_train, query_train, query_num_time_train, query_distribution_train, y_train_all, \
               x_test, time_test, query_test, query_num_time_test, query_distribution_test, y_test_all, \
               test_num_list, filtered_test_list
    else:
        print "loading data ..."
        (x_train, y_train_all, x_test, y_test_all) = cPickle.load(open(traffic_data_one_lstm_file, "rb"))
        (query_train, query_test) = cPickle.load(open(query_type_one_lstm_file, "rb"))
        (time_train, time_test) = cPickle.load(open(time_data_one_lstm_file, "rb"))
        (query_num_time_train, query_num_time_test) = cPickle.load(open(query_num_time_one_lstm_file, "rb"))
        (query_distribution_train, query_distribution_test) = cPickle.load(open(query_distribution_one_lstm_file, "rb"))
        test_num_list = cPickle.load(open(test_num_list_file, "rb"))
        filtered_test_list = cPickle.load(open(filtered_test_list_file, "rb"))
        return x_train, time_train, query_train, query_num_time_train, query_distribution_train, y_train_all, \
               x_test, time_test, query_test, query_num_time_test, query_distribution_test, y_test_all, \
               test_num_list, filtered_test_list


x_train_all, time_train, query_train, query_num_time_train, query_distribution_train, y_train, \
x_test_all, time_test, query_test, query_num_time_test, query_distribution_test, y_test, \
test_num_list, filtered_test_list = load_data()

x_train_previous = x_train_all[:, :12, :]
x_train_current = x_train_all[:, 12:, :]
x_test_previous = x_test_all[:, :12, :]
x_test_current = x_test_all[:, 12:, :]

y_train_previous = x_train_all[:, 12:, :]
y_test_previous = x_test_all[:, 12:, :]

query_num_train_previous = query_num_time_train[:, :12, 0]
query_num_test_previous = query_num_time_test[:, :12, 0]
query_num_train_previous = query_num_train_previous[:, :, None]
query_num_test_previous = query_num_test_previous[:, :, None]
query_time_train_previous = query_num_time_train[:, :12, 1]
query_time_test_previous = query_num_time_test[:, :12, 1]
query_time_train_previous = query_time_train_previous[:, :, None]
query_time_test_previous = query_time_test_previous[:, :, None]

query_num_train_current = query_num_time_train[:, 12:, 0]
query_num_test_current = query_num_time_test[:, 12:, 0]
query_num_train_current = query_num_train_current[:, :, None]
query_num_test_current = query_num_test_current[:, :, None]
query_time_train_current = query_num_time_train[:, 12:, 1]
query_time_test_current = query_num_time_test[:, 12:, 1]
query_time_train_current = query_time_train_current[:, :, None]
query_time_test_current = query_time_test_current[:, :, None]

query_distribution_train_previous = query_distribution_train[:, :12, :]
query_distribution_train_current = query_distribution_train[:, 12:, :]
query_distribution_test_previous = query_distribution_test[:, :12, :]
query_distribution_test_current = query_distribution_test[:, 12:, :]

# hyper parameters
# hyper parameters
lstm_out_dim_list = [16]
batch_size_list = [256]
epochs_list = [100]

fixed_traffic_batch = 256
fixed_traffic_epoch = 100
model_name = "LSTM3_7"

loss_weight_lambda_list = [1.0]
window_size = 12
resize_num = 5
coarse_query_type_cnt = 26

result_dict = defaultdict()
for lstm_out_dim in lstm_out_dim_list:
    for epochs in epochs_list:
        for batch_size in batch_size_list:
            for loss_weight_lambda in loss_weight_lambda_list:
                fixed_traffic_dim = lstm_out_dim*2
                print "lstm_out_dim: {}, epochs: {}, batch_size: {}, loss_weight:{}".format(lstm_out_dim, epochs,
                                                                                            batch_size,
                                                                                            loss_weight_lambda)

                first_filepath = "../weights/LSTM2_0/2-0.best.weights.dim.{}.batch.{}.hdf5".format(fixed_traffic_dim,
                                                                                                   batch_size)
                second_filepath = "../weights/LSTM2_7/LSTM2_7.tdim.{}.qdim.{}.batch.{}.hdf5".format(fixed_traffic_dim,
                                                                                                    lstm_out_dim,
                                                                                                    batch_size)
                if not os.path.exists(second_filepath):
                    print "-------------------------------------------"
                    print "No second basemodel, training a based model"
                    print "-------------------------------------------"

                # second model
                # ----------------------------------------------------------------
                # input
                traffic_input_previous = Input(shape=(window_size, resize_num), dtype='float',
                                               name="traffic_input_previous")
                traffic_input = Input(shape=(window_size, resize_num), dtype='float', name="traffic_input")
                query_num_input = Input(shape=(window_size, 1), dtype='float', name="query_num_input")
                query_time_input = Input(shape=(window_size, 1), dtype='float', name="query_time_input")
                query_distribution_input = Input(shape=(window_size, resize_num), dtype='float', name="query_distribution_input")

                query_num_input_current = Input(shape=(window_size, 1), dtype='float', name="query_num_input_current")
                query_time_input_current = Input(shape=(window_size, 1), dtype='float', name="query_time_input_current")
                query_distribution_input_current = Input(shape=(window_size, resize_num), dtype='float',
                                                         name="query_distribution_input_current")
                traffic_gt_previous = Input(shape=(window_size, resize_num), dtype='float', name="traffic_gt_previous")
                # ----------------------------------------------------------------
                # LSTM for Traffic
                traffic_encoder = LSTM(fixed_traffic_dim, activation='relu', name="traffic_encoder")
                traffic_repeatvector = RepeatVector(window_size, name="traffic_repeatvector")
                traffic_decoder = LSTM(fixed_traffic_dim, activation='relu', return_sequences=True,
                                       name="traffic_decoder")
                traffic_timedis_dense = TimeDistributed(Dense(resize_num), name="traffic_dense")
                '''
                traffic_encoder.trainable = False
                traffic_repeatvector.trainable = False
                traffic_decoder.trainable = False
                '''
                # ----------------------------------------------------------------
                traffic_encoding = traffic_encoder(traffic_input_previous)
                traffic_decoder_input = traffic_repeatvector(traffic_encoding)
                traffic_decoder_output = traffic_decoder(traffic_decoder_input)
                # ----------------------------------------------------------------
                # LSTM for Query
                query_encoder = LSTM(lstm_out_dim, activation='relu', name="query_encoder")
                query_repeatvector = RepeatVector(window_size, name="query_repeatvector")
                query_decoder = LSTM(lstm_out_dim, activation="relu", return_sequences=True, name="query_decoder")
                '''
                query_encoder.trainable = False
                query_repeatvector.trainable = False
                query_decoder.trainable = False
                '''
                # ----------------------------------------------------------------
                sequence_dense = Dense(resize_num, name="sequence_dense")
                sequence_decoder = TimeDistributed(sequence_dense, name="sequence_decoder")
               # sequence_dense.trainable = False
               # sequence_decoder.trainable = False
                # ----------------------------------------------------------------
                concat_query = concatenate([query_distribution_input, query_num_input, query_time_input])
                query_encoding = query_encoder(concat_query)
                query_decoder_input = query_repeatvector(query_encoding)
                query_decoder_output = query_decoder(query_decoder_input)

                concat_encoding = concatenate([traffic_decoder_output, query_decoder_output])
                sequence_dense_output = sequence_decoder(concat_encoding)
                # ------------------------------------------------------
                # Residual model
                traffic_residual = subtract([traffic_gt_previous, sequence_dense_output])
                residual_concat = concatenate([traffic_gt_previous, traffic_residual])
                residual_encoding = LSTM(lstm_out_dim, activation='relu')(residual_concat)
                residual_decoder_input = RepeatVector(window_size)(residual_encoding)
                residual_lstm_output = LSTM(lstm_out_dim, activation='relu', return_sequences=True)(
                    residual_decoder_input)
                residual_out = TimeDistributed(Dense(resize_num), name="residual_dense")(residual_lstm_output)
                # -------------------------------------------------------

                current_traffic_encoding = traffic_encoder(traffic_input)
                current_traffic_decoder_input = traffic_repeatvector(current_traffic_encoding)
                current_traffic_lstm_output = traffic_decoder(current_traffic_decoder_input)

                traffic_current = traffic_timedis_dense(current_traffic_lstm_output)

                '''
                current_concat_query = concatenate([query_distribution_input_current, query_time_input_current])
                current_query_encoding = query_encoder(current_concat_query)
                current_query_decoder_input = query_repeatvector(current_query_encoding)
                current_query_decoder_output = query_decoder(current_query_decoder_input)

                current_concat_encoding = concatenate([current_traffic_lstm_output, current_query_decoder_output])

                traffic_current = sequence_decoder(current_concat_encoding)
                '''
                # --------------------------------------------------------
                traffic_predict = add([traffic_current, residual_out])

                single_output_model = Model(inputs=[traffic_input_previous, traffic_input, query_distribution_input,
                                                    query_distribution_input_current, query_num_input, query_num_input_current,
                                                    query_time_input, query_time_input_current, traffic_gt_previous],
                                            outputs=[traffic_predict])
                print "our model summary"
                single_output_model.summary()

                print "frist model summary"
                first_model = load_model(first_filepath)
                first_model.summary()

                print "second model summary"
                second_model = load_model(second_filepath)
                second_model.summary()

                single_output_model.load_weights(first_filepath, by_name=True)
                single_output_model.load_weights(second_filepath, by_name=True)
                single_output_model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

                filepath1 = "../weights/{}/{}.tdim.{}.qdim.{}.batch.{}.hdf5".format(model_name, model_name,
                                                                                    fixed_traffic_dim,
                                                                                    lstm_out_dim, batch_size)
                checkpoint = ModelCheckpoint(filepath1, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
                callbacks_list = [checkpoint]
                history = single_output_model.fit([x_train_previous, x_train_current, query_distribution_train_previous,
                                                   query_distribution_train_current, query_num_train_previous, query_num_train_current,
                                                   query_time_train_previous, query_time_train_current, y_train_previous],
                                                  [y_train], epochs=epochs , batch_size=batch_size, verbose=2,
                                                  callbacks=callbacks_list,
                                                  validation_data=([x_test_previous, x_test_current, query_distribution_test_previous,
                                                   query_distribution_test_current, query_num_test_previous, query_num_test_current,
                                                   query_time_test_previous, query_time_test_current, y_test_previous], [y_test]))

                metric_list = ['mean_squared_error', 'mean_absolute_error']
                for metric in metric_list:
                    fig = plt.figure()
                    plt.plot(history.history[metric])
                    plt.plot(history.history['val_' + metric])
                    plt.title(metric)
                    plt.ylabel(metric)
                    plt.xlabel('epoch')
                    plt.legend(['train', 'test'], loc='upper right')
                    fig.savefig(
                        '../fig/{}/{}.tdim.{}.qdim.{}.batch.{}'.format(model_name, model_name, fixed_traffic_dim,
                                                                       lstm_out_dim, batch_size) + metric + '.png')
