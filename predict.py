import os
import sys
import json
import shutil
import pickle
import logging
import data_helper
import numpy as np
import pandas as pd
import tensorflow as tf
import csv
from text_cnn_rnn import TextCNNRNN

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
logging.getLogger().setLevel(logging.INFO)


def load_trained_params(trained_dir):
    params = json.loads(open(trained_dir + 'trained_parameters.json').read())
    words_index = json.loads(open(trained_dir + 'words_index.json').read())
    labels = json.loads(open(trained_dir + 'labels.json').read())

    with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
        fetched_embedding = pickle.load(input_file)
    embedding_mat = np.array(fetched_embedding, dtype=np.float32)
    return params, words_index, labels, embedding_mat


def load_test_data(test_file):
    df_test = pd.read_csv(test_file, compression='zip')
    select = ['text']
    x_test = df_test[select[0]].apply(lambda x: data_helper.clean_str(x).split(' ')).tolist()

    return x_test, df_test


def predict_unseen_data(trained_timestamp=None, test_file=None):
    if trained_timestamp is None:
        trained_timestamp = sys.argv[1]
    if test_file is None:
        test_file = sys.argv[2]

    trained_dir = './trained_results_' + trained_timestamp + '/'
    checkpoint_dir = './checkpoints_' + trained_timestamp + '/'

    params, words_index, labels, embedding_mat = load_trained_params(trained_dir)
    x_test, df_test = load_test_data(test_file)
    x_test = data_helper.pad_sentences(x_test, forced_sequence_length=params['sequence_length'])
    x_test = list(map(lambda x: list(map(lambda y: words_index[y] if y in words_index else 0, x)), x_test))
    x_test = np.array(x_test)

    predicted_dir = './predicted_results_' + trained_timestamp + '/'
    if os.path.exists(predicted_dir):
        shutil.rmtree(predicted_dir)
    os.makedirs(predicted_dir)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn_rnn = TextCNNRNN(
                embedding_mat=embedding_mat,
                non_static=params['non_static'],
                hidden_unit=params['hidden_unit'],
                sequence_length=len(x_test[0]),
                max_pool_size=params['max_pool_size'],
                filter_sizes=map(int, params['filter_sizes'].split(",")),
                num_filters=params['num_filters'],
                num_classes=len(labels),
                embedding_size=params['embedding_dim'],
                l2_reg_lambda=params['l2_reg_lambda'])

            def real_len(batches):
                return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]

            def predict_step(x_batch):
                feed_dict = {
                    cnn_rnn.input_x: x_batch,
                    cnn_rnn.dropout_keep_prob: 1.0,
                    cnn_rnn.batch_size: len(x_batch),
                    cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
                    cnn_rnn.real_len: real_len(x_batch),
                }
                return sess.run(cnn_rnn.probabilities, feed_dict)

            checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_file)
            logging.critical('{} has been loaded'.format(checkpoint_file))

            batches = data_helper.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)

            probabilities = []
            for x_batch in batches:
                batch_probabilities = predict_step(x_batch)
                for probability in batch_probabilities:
                    probabilities.append(probability)
            probabilities = np.array(probabilities)

            for idx, label in enumerate(labels):
                df_test[label] = probabilities[:, idx]

            columns = ["id", "EAP", "HPL", "MWS"]
            df_test.to_csv(predicted_dir + 'predictions.csv', index=False, columns=columns, quoting=csv.QUOTE_NONNUMERIC)

            logging.critical('Prediction is complete, all files have been saved: {}'.format(predicted_dir))


if __name__ == '__main__':
    predict_unseen_data(trained_timestamp='1509709655', test_file='./data/test.zip')
