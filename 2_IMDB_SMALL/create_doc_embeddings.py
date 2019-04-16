import utils
import numpy as np


def concat_embeddings():
    emb1_name = 'dbow_bert_ver2_500_200'
    emb2_name = 'bert_doc_embeddings_100'
    emb1 = utils.load_doc_embeddings(emb1_name)
    emb2 = utils.load_doc_embeddings(emb2_name)
    emb = utils.concat_embeddings([emb1, emb2])
    emb = utils.normalize_embeddings(emb)
    utils.save_doc_embeddings(emb, emb1_name + ' + ' + emb2_name)


if __name__ == '__main__':
    concat_embeddings()
