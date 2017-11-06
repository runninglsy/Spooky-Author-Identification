import re
import pandas as pd
import numpy as np
import logging
from collections import Counter
import itertools
import gensim

logging.getLogger().setLevel(logging.INFO)


def clean_str(s):
    s = re.sub(r"[^A-Za-z0-9:(),!?'`]", " ", s)
    s = re.sub(r":", " : ", s)
    s = re.sub(r"'s", " 's", s)  # Bob's -> Bob 's
    s = re.sub(r"'ve", " 've", s)  # I've -> I 've
    s = re.sub(r"can't", "can n't", s)
    s = re.sub(r"n't", " n't", s)  # don't -> do n't
    s = re.sub(r"'re", " 're", s)  # you're -> you 're
    s = re.sub(r"'d", " 'd", s)  # I'd -> I 'd
    s = re.sub(r"'ll", " 'll", s)  # I'll -> I 'll
    s = re.sub(r"['`\"]", " ", s)  # remove `'"
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " ( ", s)
    s = re.sub(r"\)", " ) ", s)
    s = re.sub(r"\?", " ? ", s)
    s = re.sub(r"\s{2,}", " ", s)  # remove extra spaces
    return s.strip().lower()


def build_vocab(sentences, padding_word="$"):
    word_counter = Counter(itertools.chain(*sentences))
    vocabulary_lst = [cnt[0] for cnt in word_counter.most_common()]
    idx_padding_word = vocabulary_lst.index(padding_word)
    vocabulary_lst[0], vocabulary_lst[idx_padding_word] = vocabulary_lst[idx_padding_word], vocabulary_lst[0]

    vocabulary_dict = dict([(word, idx) for idx, word in enumerate(vocabulary_lst)])
    return vocabulary_lst, vocabulary_dict


def pad_sentences(sentences, padding_word="$", forced_sequence_length=None):
    """Pad sentences during training or prediction"""
    if forced_sequence_length is None:  # Train
        sequence_length = (max(len(x) for x in sentences))
        if sequence_length >= 30:
            sequence_length = 30
    else:  # Prediction
        logging.critical('This is prediction, reading the trained sequence length')
        sequence_length = forced_sequence_length
    logging.critical('The maximum length of sentence is {}'.format(sequence_length))

    padded_sentences = []
    cut_cnt = 0
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)

        if num_padding < 0:  # cut off the sentence if it is longer than the sequence length
            cut_cnt += 1
            padded_sentence = sentence[0:sequence_length]
        else:
            padded_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(padded_sentence)

    logging.info('{} sentence(s) has to be cut off'
                 ' because it is longer than trained sequence length'.format(cut_cnt))
    return padded_sentences


# def load_embeddings(vocabulary_lst, embedding_dim):
#     word_embeddings = {}
#     for word in vocabulary_lst:
#         word_embeddings[word] = np.random.uniform(-0.25, 0.25, embedding_dim)
#     return word_embeddings


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_data(train_file, test_file, embedding_dim):
    # read train data from .zip file
    train_data = pd.read_csv(train_file, compression="zip")
    selected = ["text", "author"]
    non_selected = list(set(train_data.columns) - set(selected))

    # drop unuseful column and incomplete line
    train_data = train_data.drop(non_selected, axis=1)
    train_data = train_data.dropna(axis=0, how='any')
    train_data = train_data.reindex(np.random.permutation(train_data.index))

    # map each label(string type) to an one-hot vector
    labels = sorted(list(set(train_data[selected[1]].values)))
    n_labels = len(labels)
    one_hot_mat = np.identity(n_labels, dtype=int)
    label_dict = dict(zip(labels, one_hot_mat))

    # convert each x(sentence) to an list of words
    x_raw = train_data[selected[0]].apply(lambda x: clean_str(x).split(" ")).tolist()
    # convert each y(label) to an one-hot vector
    y_raw = train_data[selected[1]].apply(lambda x: label_dict[x]).tolist()

    # add test data for word2vec training
    test_data = pd.read_csv(test_file, compression="zip")
    test_x_raw = test_data[selected[0]].apply(lambda x: clean_str(x).split(" ")).tolist()

    # padding to make all sentences have the same length.
    x_raw = pad_sentences(x_raw)

    test_x_raw = pad_sentences(test_x_raw)

    # create vocabulary
    vocabulary_lst, vocabulary_dict = build_vocab(x_raw + test_x_raw)

    # word2vec
    model = gensim.models.Word2Vec(x_raw + test_x_raw, min_count=0, workers=16, size=embedding_dim)

    # convert representation of x from list of words to list of numbers
    x_lst = np.array([[vocabulary_dict[word] for word in sentence] for sentence in x_raw])

    y_lst = np.array(y_raw)

    return x_lst, y_lst, vocabulary_lst, vocabulary_dict, labels, model

