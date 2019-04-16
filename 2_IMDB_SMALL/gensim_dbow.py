import utils
import gensim.models.doc2vec
import multiprocessing
import logging
import configparser
from collections import namedtuple
from random import shuffle
from gensim.models import Doc2Vec
from dataset_imdb import Doc


def train(doc_embedding_size, negative_sample_size, epochs):

    print('doc_embedding_size:', doc_embedding_size)
    print('negative_sample_size:', negative_sample_size)
    print('epochs:', epochs)

    logging.getLogger().setLevel(logging.DEBUG)

    all_docs = utils.load('all_docs')
    alldocs = []
    corpus_size = len(all_docs)

    ImdbDocument = namedtuple('ImdbDocument', 'words tags')

    for i in range(corpus_size):
        words = all_docs[i].words
        tags = [i]
        alldocs.append(ImdbDocument(words, tags))

    print('docs size:', len(alldocs))

    doc_list = alldocs[:]
    shuffle(doc_list)

    cores = multiprocessing.cpu_count()
    assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

    model = Doc2Vec(dm=0, vector_size=doc_embedding_size, negative=negative_sample_size, hs=0, min_count=2, sample=0,
                    epochs=epochs, workers=cores)

    # Build corpus
    model.build_vocab(alldocs)
    print("%s vocabulary scanned & state initialized" % model)
    print("vocab size:", len(model.wv.vocab))
    print("docvecs size:", len(model.docvecs))

    # Train
    print("Training %s" % model)
    model.train(doc_list, total_examples=len(doc_list), epochs=model.epochs)

    # Save
    emb = []
    for i in range(corpus_size):
        emb.append(model.docvecs[i])

    emb = utils.normalize_embeddings(emb)

    utils.save_doc_embeddings(emb, 'gensim_dbow', negative_size=negative_sample_size)


def main():
    # Load config
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Normal params
    doc_embedding_size = int(config['data']['doc_embedding_size'])
    epochs = int(config['gensim dbow']['epochs'])

    # List params
    negative_sample_size = str(config['gensim dbow']['negative_sample_size'])
    negative_sample_list = [int(x) for x in negative_sample_size.split(',')]

    # train
    for i in range(len(negative_sample_list)):
        train(doc_embedding_size=doc_embedding_size, negative_sample_size=negative_sample_list[i], epochs=epochs)


if __name__ == '__main__':
    main()
