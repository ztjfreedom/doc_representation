import tensorflow as tf
import numpy as np
import os
import utils
import math
import configparser
import dataset
from tensorflow.python.framework import ops
from random import shuffle


def create_train_data():
    train_data_file_name = 'memory_dbow_train'
    label_data_file_name = 'memory_dbow_label'

    if utils.exists(train_data_file_name) and utils.exists(label_data_file_name):
        train_data = utils.load(train_data_file_name)
        label_data = utils.load(label_data_file_name)
    else:
        all_docs = utils.load('all_docs')
        pairs = [([doc.tag, sentence_num], word_num)
                 for doc in all_docs
                 for sentence_num in doc.detail_sentences_nums
                 for word_num in doc.title_words_nums]
        detail_pairs = [([doc.tag, sentence_num], word_num)
                        for doc in all_docs
                        for sentence_num in doc.detail_sentences_nums
                        for word_num in doc.detail_words_nums]
        pairs.extend(detail_pairs)

        print('Shuffling train data set, size:', len(pairs))
        shuffle(pairs)

        train_data, label_data = [list(pair) for pair in zip(*pairs)]

        train_data = np.array(train_data)
        label_data = np.transpose(np.array([label_data]))

        utils.save(train_data, train_data_file_name)
        utils.save(label_data, label_data_file_name)

    return train_data, label_data


def train(gpu_no, show_loss, train_data, label_data, word_embedding_size, doc_embedding_size,
          batch_size, negative_sample_size, epochs):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_no)

    print('word_embedding_size', word_embedding_size)
    print('doc_embedding_size', doc_embedding_size)
    print('batch_size', batch_size)
    print('negative_sample_size', negative_sample_size)
    print('epochs:', epochs)

    # Init
    ops.reset_default_graph()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Load
    print('Loading pre processed data')
    all_docs = utils.load('all_docs')
    word_dictionary = utils.load('word_dictionary')
    bert_title_embeddings = utils.load('bert_title_embeddings')
    bert_detail_sentence_embeddings = utils.load('bert_detail_sentence_embeddings')

    docs_size = len(all_docs)
    vocabulary_size = len(word_dictionary)
    train_set_size = len(train_data)
    final_embedding_size = doc_embedding_size

    print('vocabulary_size:', vocabulary_size)
    print('final_embedding_size:', final_embedding_size)
    print('train_set_size:', train_set_size)

    print('Creating model')

    # Define Embeddings:
    with tf.name_scope('embeddings'):
        detail_sentence_embeddings = tf.constant(bert_detail_sentence_embeddings)
        title_embeddings = tf.constant(bert_title_embeddings)
        doc_embeddings = tf.Variable(tf.random_uniform([docs_size, doc_embedding_size], -1.0, 1.0))
        title_weights = tf.Variable(tf.random_uniform([docs_size, 1], 0.0, 1.0))

    # NCE loss parameters
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, final_embedding_size],
                                                  stddev=1.0 / np.sqrt(final_embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Create data/target placeholders
    x_inputs = tf.placeholder(tf.int32, shape=[None, 2])
    y_target = tf.placeholder(tf.int32, shape=[None, 1])

    # Lookup the word embedding
    # Add together element embeddings in window:
    # Concat all embeddings

    doc_indices = tf.slice(x_inputs, [0, 0], [batch_size, 1])
    sentence_indices = tf.slice(x_inputs, [0, 1], [batch_size, 1])

    doc_embed = tf.squeeze(tf.nn.embedding_lookup(doc_embeddings, doc_indices), axis=1)
    sentence_embed = tf.squeeze(tf.nn.embedding_lookup(detail_sentence_embeddings, sentence_indices), axis=1)
    title_embed = tf.squeeze(tf.nn.embedding_lookup(title_embeddings, doc_indices), axis=1)

    title_weight = tf.squeeze(tf.nn.embedding_lookup(title_weights, doc_indices), axis=1)
    title_embed_weighted = tf.math.multiply(title_embed, title_weight)

    final_embed = (title_embed_weighted + sentence_embed + doc_embed) / 3

    # Get loss from prediction
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, y_target, final_embed,
                                             negative_sample_size, vocabulary_size))

    # Create optimizer
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
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
                batch_train, batch_label = dataset.generate_batch_data(train_data, label_data,
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

        title_weights = sess.run(title_weights)
        title_embeddings_weighted = np.multiply(bert_title_embeddings, title_weights)
        detail_emb = utils.mean_embeddings([title_embeddings_weighted, doc_embeddings])
        detail_emb_norm = utils.normalize_embeddings(detail_emb)
        utils.save_doc_embeddings(detail_emb_norm, 'memory_dbow_detail',
                                  batch_size=batch_size, negative_size=negative_sample_size)

        emb = utils.concat_embeddings([bert_title_embeddings, detail_emb_norm])
        utils.save_doc_embeddings(emb, 'memory_dbow',
                                  batch_size=batch_size, negative_size=negative_sample_size)


def main():
    # Load config
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Normal params
    word_embedding_size = int(config['data']['word_embedding_size'])
    doc_embedding_size = int(config['data']['doc_embedding_size'])
    epochs = int(config['memory dbow']['epochs'])

    # List params
    batch_size = str(config['memory dbow']['batch_size'])
    negative_sample_size = str(config['memory dbow']['negative_sample_size'])

    batch_list = [int(x) for x in batch_size.split(',')]
    negative_sample_list = [int(x) for x in negative_sample_size.split(',')]
    assert len(batch_list) == len(negative_sample_list)

    # train data
    print('Creating train data')
    train_data, label_data = create_train_data()

    for i in range(len(batch_list)):
        train(gpu_no=0, show_loss=True,
              train_data=train_data, label_data=label_data,
              word_embedding_size=word_embedding_size, doc_embedding_size=doc_embedding_size,
              batch_size=batch_list[i], negative_sample_size=negative_sample_list[i], epochs=epochs)


if __name__ == '__main__':
    main()
