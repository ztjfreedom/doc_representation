import utils


def concat_embeddings(emb1_name, emb2_name):
    emb1 = utils.load_doc_embeddings(emb1_name)
    emb2 = utils.load_doc_embeddings(emb2_name)
    emb = utils.concat_embeddings([emb1, emb2])
    emb = utils.normalize_embeddings(emb)
    utils.save_doc_embeddings(emb, emb1_name + ' + ' + emb2_name)


if __name__ == '__main__':
    concat_embeddings('bert_title_embeddings_100', 'dm_bert_5_500_5')
