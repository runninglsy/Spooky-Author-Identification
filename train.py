import os
import sys
import json
import time
import shutil
import pickle
import logging
import data_helper
import numpy as np
import pandas as pd
from text_cnn_rnn import TextCNNRNN
from sklearn.model_selection import train_test_split
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
logging.getLogger().setLevel(logging.INFO)


def train_cnn_rnn(input_file=None, training_config=None):
    if input_file is None:
        input_file = sys.argv[1]
    if training_config is None:
        training_config = sys.argv[2]

    x_lst, y_lst, vocabulary_lst, vocabulary_dict, labels = data_helper.load_data(input_file)
    params = json.loads(open(training_config).read())

    word_embeddings = data_helper.load_embeddings(vocabulary_lst)
    embedding_mat = [word_embeddings[word] for _, word in enumerate(vocabulary_lst)]
    embedding_mat = np.array(embedding_mat, dtype=np.float32)

    x_, x_test, y_, y_test = train_test_split(x_lst, y_lst, test_size=0.2)
    x_train, x_dev, y_train, y_dev = train_test_split(x_, y_, test_size=0.1)

    logging.info('number of samples: x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
    logging.info('number of samples: y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

    # Create a directory, everything related to the training will be saved in this directory
    timestamp = str(int(time.time()))
    trained_dir = './trained_results_' + timestamp + '/'
    if os.path.exists(trained_dir):
        shutil.rmtree(trained_dir)
    os.makedirs(trained_dir)

    graph = tf.Graph()
    with graph.as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn_rnn = TextCNNRNN(
                embedding_mat=embedding_mat,
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                non_static=params['non_static'],
                hidden_unit=params['hidden_unit'],
                max_pool_size=params['max_pool_size'],
                filter_sizes=map(int, params['filter_sizes'].split(",")),
                num_filters=params['num_filters'],
                embedding_size=params['embedding_dim'],
                l2_reg_lambda=params['l2_reg_lambda'])

            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.RMSPropOptimizer(1e-3, decay=0.9)
            grads_and_vars = optimizer.compute_gradients(cnn_rnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Checkpoint files will be saved in this directory during training
            checkpoint_dir = './checkpoints_' + timestamp + '/'
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            os.makedirs(checkpoint_dir)
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

            def real_len(batches):
                return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]

            def train_step(x_batch, y_batch):
                feed_dict = {
                    cnn_rnn.input_x: x_batch,
                    cnn_rnn.input_y: y_batch,
                    cnn_rnn.dropout_keep_prob: params['dropout_keep_prob'],
                    cnn_rnn.batch_size: len(x_batch),
                    cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
                    cnn_rnn.real_len: real_len(x_batch),
                }
                _, step, loss, accuracy = sess.run([train_op, global_step, cnn_rnn.loss, cnn_rnn.accuracy], feed_dict)

            def dev_step(x_batch, y_batch):
                feed_dict = {
                    cnn_rnn.input_x: x_batch,
                    cnn_rnn.input_y: y_batch,
                    cnn_rnn.dropout_keep_prob: 1.0,
                    cnn_rnn.batch_size: len(x_batch),
                    cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
                    cnn_rnn.real_len: real_len(x_batch),
                }
                step, loss, accuracy, num_correct, predictions = sess.run(
                    [global_step, cnn_rnn.loss, cnn_rnn.accuracy, cnn_rnn.num_correct, cnn_rnn.predictions], feed_dict)
                return accuracy, loss, num_correct, predictions

            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            # Training starts here
            train_batches = data_helper.batch_iter(list(zip(x_train, y_train)), params['batch_size'],
                                                   params['num_epochs'])
            best_accuracy, best_at_step, best_loss = 0, 0, 100000000000000000
            path = None
            # Train the model with x_train and y_train
            for train_batch in train_batches:
                x_train_batch, y_train_batch = zip(*train_batch)
                train_step(x_train_batch, y_train_batch)
                current_step = tf.train.global_step(sess, global_step)

                # Evaluate the model with x_dev and y_dev
                if current_step % params['evaluate_every'] == 0:
                    dev_batches = data_helper.batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 1)

                    total_dev_correct = 0
                    loss_dev_lst = []
                    for dev_batch in dev_batches:
                        x_dev_batch, y_dev_batch = zip(*dev_batch)
                        acc, loss, num_dev_correct, predictions = dev_step(x_dev_batch, y_dev_batch)
                        loss_dev_lst.append(loss)
                        total_dev_correct += num_dev_correct
                    accuracy = float(total_dev_correct) / len(y_dev)
                    logging.info('Accuracy on dev set: {}'.format(accuracy))
                    logging.info('Loss on dev set: {}'.format(np.mean(loss_dev_lst)))

                    loss_dev = np.mean(loss_dev_lst)
                    if loss_dev < best_loss:
                        best_loss, best_at_step = loss_dev, current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        logging.critical('Best loss {} at step {}'.format(best_loss, best_at_step))
                        logging.critical('Saved model {} at step {}'.format(path, best_at_step))
            logging.critical('Training is complete, testing the best model on x_test and y_test')

            # Evaluate x_test and y_test
            saver.restore(sess, path)
            test_batches = data_helper.batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1, shuffle=False)
            total_test_correct = 0
            loss_test_lst = []
            for test_batch in test_batches:
                x_test_batch, y_test_batch = zip(*test_batch)
                acc, loss, num_test_correct, predictions = dev_step(x_test_batch, y_test_batch)
                loss_test_lst.append(loss)
                total_test_correct += int(num_test_correct)
            logging.critical('Accuracy on test set: {}'.format(float(total_test_correct) / len(y_test)))
            logging.critical('Loss on test set: {}'.format(np.mean(loss_test_lst)))

    # Save trained parameters and files since predict.py needs them
    with open(trained_dir + 'words_index.json', 'w') as outfile:
        json.dump(vocabulary_dict, outfile, indent=4, ensure_ascii=False)
    with open(trained_dir + 'embeddings.pickle', 'wb') as outfile:
        pickle.dump(embedding_mat, outfile, pickle.HIGHEST_PROTOCOL)
    with open(trained_dir + 'labels.json', 'w') as outfile:
        json.dump(labels, outfile, indent=4, ensure_ascii=False)

    # os.renames(path, trained_dir + 'best_model.ckpt')
    # os.renames(path + '.meta', trained_dir + 'best_model.meta')
    # shutil.rmtree(checkpoint_dir)
    # logging.critical('{} has been removed'.format(checkpoint_dir))

    params['sequence_length'] = x_train.shape[1]
    with open(trained_dir + 'trained_parameters.json', 'w') as outfile:
        json.dump(params, outfile, indent=4, sort_keys=True, ensure_ascii=False)


if __name__ == '__main__':
    train_cnn_rnn("./data/train.zip", "./training_config.json")

