import tensorflow as tf
import numpy as np
import os
import utils
import math
import configparser
import dataset_imdb
from tensorflow.python.framework import ops
from random import shuffle


proj_name = 'dm_bert'


def create_train_data(window_size, is_cache=False):
    if len(proj_name) == 0:
        train_name = 'train_' + str(window_size)
        label_name = 'label_' + str(window_size)
    else:
        train_name = 'train_' + str(window_size) + '_' + proj_name
        label_name = 'label_' + str(window_size) + '_' + proj_name

    if is_cache and utils.exists(train_name) and utils.exists(label_name):
        train_data = utils.load(train_name)
        label_data = utils.load(label_name)
    else:
        all_docs = utils.load('all_docs')
        for doc in all_docs:
            if len(doc.numbers) <= window_size:
                nums_padded = [1] * (window_size + 1 - len(doc.numbers))
                nums_padded.extend(doc.numbers)
                doc.numbers = nums_padded

        train_label_pairs = [(doc.numbers[i: i + window_size] + [doc.tag],
                              doc.numbers[i + window_size])
                             for doc in all_docs
                             for i in range(0, len(doc.numbers) - window_size)]

        shuffle(train_label_pairs)

        train_data, label_data = [list(pair) for pair in zip(*train_label_pairs)]
        train_data = np.array(train_data)
        label_data = np.transpose(np.array([label_data]))

        if is_cache:
            utils.save(train_data, train_name)
            utils.save(label_data, label_name)

    return train_data, label_data


def train(gpu_no, show_loss, train_data, label_data, window_size, word_embedding_size, doc_embedding_size,
          batch_size, negative_sample_size, is_concat, epochs):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_no)

    print('window_size', window_size)
    print('word_embedding_size', word_embedding_size)
    print('doc_embedding_size', doc_embedding_size)
    print('batch_size', batch_size)
    print('negative_sample_size', negative_sample_size)
    print('is_concat', is_concat)
    print('epochs:', epochs)

    # Init
    ops.reset_default_graph()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Load
    print('Loading pre processed data')
    all_docs = utils.load('all_docs')
    word_dictionary = utils.load('word_dictionary')
    bert_word_embeddings = utils.load('bert_word_embeddings_100')

    docs_size = len(all_docs)
    vocabulary_size = len(word_dictionary)
    train_set_size = len(train_data)
    if is_concat:
        final_embedding_size = word_embedding_size * window_size + doc_embedding_size
    else:
        final_embedding_size = doc_embedding_size

    print('vocabulary_size:', vocabulary_size)
    print('final_embedding_size:', final_embedding_size)
    print('train_set_size:', train_set_size)

    print('Creating model')

    # Define Embeddings:
    with tf.name_scope('embeddings'):
        special_word_embeddings = tf.Variable(tf.random_uniform([2, word_embedding_size], -1.0, 1.0))
        word_embeddings = tf.concat([special_word_embeddings, tf.constant(bert_word_embeddings[2:])], axis=0)

        # word_embeddings = tf.constant(bert_word_embeddings)
        doc_embeddings = tf.Variable(tf.random_uniform([docs_size, doc_embedding_size], -1.0, 1.0))

    # NCE loss parameters
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, final_embedding_size],
                                                  stddev=1.0 / np.sqrt(final_embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Create data/target placeholders
    x_inputs = tf.placeholder(tf.int32, shape=[None, window_size + 1])  # plus 1 for doc index
    y_target = tf.placeholder(tf.int32, shape=[None, 1])

    # Lookup the word embedding
    # Add together element embeddings in window:
    # Concat all embeddings
    if is_concat:
        word_embed = [tf.nn.embedding_lookup(word_embeddings, x_inputs[:, element]) for element in range(window_size)]
        doc_indices = tf.slice(x_inputs, [0, window_size], [batch_size, 1])
        doc_embed = tf.nn.embedding_lookup(doc_embeddings, doc_indices)
        final_embed = tf.concat([*word_embed, tf.squeeze(doc_embed, axis=1)], 1)
    else:
        word_embed = tf.zeros([batch_size, word_embedding_size])
        for element in range(window_size):
            word_embed += tf.nn.embedding_lookup(word_embeddings, x_inputs[:, element])
        doc_indices = tf.slice(x_inputs, [0, window_size], [batch_size, 1])
        doc_embed = tf.squeeze(tf.nn.embedding_lookup(doc_embeddings, doc_indices), axis=1)
        final_embed = (word_embed + doc_embed) / (window_size + 1)

    # Get loss from prediction
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, y_target, final_embed,
                                             negative_sample_size, vocabulary_size))

    # Create optimizer
    optimizer = tf.train.AdamOptimizer()
    train_step = optimizer.minimize(loss)

    # Add variable initializer.
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        print('Starting training')
        generations = math.ceil(train_set_size / batch_size)
        for epoch in range(epochs):
            for generation in range(generations):
                # Generate training data
                batch_train, batch_label = dataset_imdb.generate_batch_data(train_data, label_data,
                                                                            batch_size, generation)

                # Run the train step
                feed_dict = {x_inputs: batch_train, y_target: batch_label}
                sess.run(train_step, feed_dict=feed_dict)

                # Print the loss
                if show_loss and (generation + 1) == generations:
                    loss_val = sess.run(loss, feed_dict=feed_dict)
                    print('Loss at epoch {} : {}'.format(epoch, loss_val))

        print('Saving model')
        doc_embeddings = sess.run(doc_embeddings)

        # Norm
        doc_embeddings = utils.normalize_embeddings(doc_embeddings)
        utils.save_doc_embeddings(doc_embeddings, proj_name, is_concat=is_concat, window_size=window_size,
                                  batch_size=batch_size, negative_size=negative_sample_size)


def main():
    # Load config
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Normal params
    word_embedding_size = int(config['data']['word_embedding_size'])
    doc_embedding_size = int(config['data']['doc_embedding_size'])
    is_concat = True if config['dm bert']['is_concat'] == 'True' else False
    epochs = int(config['dm bert']['epochs'])

    # List params
    window_size = str(config['dm bert']['window_size'])
    batch_size = str(config['dm bert']['batch_size'])
    negative_sample_size = str(config['dm bert']['negative_sample_size'])

    window_size_list = [int(x) for x in window_size.split(',')]
    batch_list = [int(x) for x in batch_size.split(',')]
    negative_sample_list = [int(x) for x in negative_sample_size.split(',')]
    assert len(batch_list) == len(negative_sample_list) == len(window_size_list)

    # train
    for i in range(len(batch_list)):
        # train data
        print('Creating train data')
        train_data, label_data = create_train_data(window_size_list[i], False)

        train(gpu_no=0, show_loss=True,
              train_data=train_data, label_data=label_data,
              word_embedding_size=word_embedding_size, doc_embedding_size=doc_embedding_size,
              window_size=window_size_list[i], batch_size=batch_list[i], negative_sample_size=negative_sample_list[i],
              is_concat=is_concat, epochs=epochs)


if __name__ == '__main__':
    main()
