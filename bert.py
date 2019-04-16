import numpy as np
import utils
from bert_serving.client import BertClient
from dataset import Doc


def show_words_length_info(word_dictionary):
    print('dict length:', len(word_dictionary))
    words = [word for word in word_dictionary.keys()]
    words_len = [len(word) for word in words]
    print('max length of words:', max(words_len))
    print('avg length of words:', sum(words_len) / len(words))
    return words


def show_summaries_length_info(all_docs):
    titles = [doc.title for doc in all_docs]
    titles_len = [len(title) for title in titles]
    print('max length of titles:', max(titles_len))
    print('avg length of titles:', sum(titles_len) / len(titles))
    return titles


def show_details_length_info(all_docs):
    details = [doc.detail for doc in all_docs]
    details_len = [len(detail) for detail in details]
    print('max length of details:', max(details_len))
    print('avg length of details:', sum(details_len) / len(details))
    return details


def show_details_sentences_length_info(all_docs):
    details_sentences = [x for doc in all_docs for x in doc.detail_sentences]
    details_sentences_len = [len(x) for x in details_sentences]
    print('sentences num:', len(details_sentences))
    print('max length of details sentences:', max(details_sentences_len))
    print('avg length of details sentences:', sum(details_sentences_len) / len(details_sentences))
    return details_sentences


def show_all_length_info():
    # Load
    word_dictionary = utils.load('word_dictionary')
    all_docs = utils.load('all_docs')
    print('docs num:', len(all_docs))

    # Show length info
    show_words_length_info(word_dictionary)
    show_summaries_length_info(all_docs)
    show_details_length_info(all_docs)
    show_details_sentences_length_info(all_docs)


def create_emb(seq_list, save_file_name, bert_service_ip=None, reduced_size=100, normalize=False):
    # Bert
    print('Bert converting...')
    if bert_service_ip is None:
        bc = BertClient()
    else:
        bc = BertClient(ip=bert_service_ip)
    vecs = bc.encode(seq_list)
    print('vecs type:', type(vecs))
    print('vecs shape:', vecs.shape)

    if normalize:
        vecs = utils.normalize_embeddings(vecs)

    # Save
    utils.save(vecs, save_file_name)

    # PCA
    vecs = utils.reduce_dim(vecs, reduced_size)

    if normalize:
        vecs = utils.normalize_embeddings(vecs)

    utils.save(vecs, save_file_name + '_' + str(reduced_size))


def create_word_emb(bert_service_ip=None, reduced_size=100, normalize=False):
    # Load dict
    word_dictionary = utils.load('word_dictionary')

    # Show length info
    words = show_words_length_info(word_dictionary)

    # Bert
    print('Bert converting...')
    if bert_service_ip is None:
        bc = BertClient()
    else:
        bc = BertClient(ip=bert_service_ip)
    vecs = bc.encode(words)
    print('vecs type:', type(vecs))
    print('vecs shape:', vecs.shape)

    if normalize:
        vecs = utils.normalize_embeddings(vecs)

    # Save
    utils.save(vecs, 'bert_word_embeddings')

    # PCA
    vecs = utils.reduce_dim(vecs, reduced_size)

    if normalize:
        vecs = utils.normalize_embeddings(vecs)

    utils.save(vecs, 'bert_word_embeddings_' + str(reduced_size))


def test_word_emb(sample_words):
    word_embeddings = utils.load('bert_word_embeddings')
    word_dictionary = utils.load('word_dictionary')
    word_dictionary_rev = utils.load('word_dictionary_rev')

    for sample_word in sample_words:
        embed = word_embeddings[word_dictionary[sample_word]]
        similarities = []
        for word_embedding in word_embeddings:
            similarities.append(utils.cosine_dist(np.array(embed), np.array(word_embedding)))
        array = np.array(similarities)
        indices = array.argsort()[-10:][::-1]
        print('nearest to', sample_word, ':')
        print([word_dictionary_rev[index] for index in indices])


def create_summary_emb(bert_service_ip=None, reduced_size=100, normalize=False):
    # Load docs
    all_docs = utils.load('all_docs')
    print('docs num:', len(all_docs))

    # Show length info
    titles = show_summaries_length_info(all_docs)

    # Bert
    print('Bert converting...')
    if bert_service_ip is None:
        bc = BertClient()
    else:
        bc = BertClient(ip=bert_service_ip)

    print('Converting title')
    vecs = bc.encode(titles)
    print('title_vecs shape:', vecs.shape)

    if normalize:
        vecs = utils.normalize_embeddings(vecs)

    utils.save(vecs, 'bert_title_embeddings')

    # PCA
    vecs = utils.reduce_dim(vecs, reduced_size)

    if normalize:
        vecs = utils.normalize_embeddings(vecs)

    utils.save(vecs, 'bert_title_embeddings_' + str(reduced_size))


def create_detail_emb(bert_service_ip=None, reduced_size=100, normalize=False):
    # Load docs
    all_docs = utils.load('all_docs')
    print('docs num:', len(all_docs))

    # Show length info
    details = show_details_length_info(all_docs)

    # Bert
    print('Bert converting...')
    if bert_service_ip is None:
        bc = BertClient()
    else:
        bc = BertClient(ip=bert_service_ip)

    print('Converting detail')
    vecs = bc.encode(details)
    print('detail_vecs shape:', vecs.shape)

    if normalize:
        vecs = utils.normalize_embeddings(vecs)

    utils.save(vecs, 'bert_detail_embeddings')

    # PCA
    vecs = utils.reduce_dim(vecs, reduced_size)

    if normalize:
        vecs = utils.normalize_embeddings(vecs)

    utils.save(vecs, 'bert_detail_embeddings_' + str(reduced_size))


# def create_detail_sentence_emb(bert_service_ip=None, doc_embedding_size=100, normalize=False, reduce_dim=True):
#     # Load docs
#     all_docs = utils.load('all_docs')
#     print('docs num:', len(all_docs))
#
#     # Show length info
#     details_sentences = show_details_sentences_length_info(all_docs)
#
#     # Bert
#     print('Bert converting...')
#     if bert_service_ip is None:
#         bc = BertClient()
#     else:
#         bc = BertClient(ip=bert_service_ip)
#
#     print('Converting detail sentence')
#     vecs = bc.encode(details_sentences)
#     print('detail_sentence_vecs shape:', vecs.shape)
#
#     # PCA
#     if reduce_dim:
#         vecs = utils.reduce_dim(vecs, doc_embedding_size)
#
#     # Normalize
#     if normalize:
#         vecs = utils.normalize_embeddings(vecs)
#
#     utils.save(vecs, 'bert_detail_sentence_embeddings')


# def avg_detail_sentence_emb():
#     all_docs = utils.load('all_docs')
#     sentence_embeddings = utils.load('bert_detail_sentence_embeddings')
#     avg_embs = []
#     for doc in all_docs:
#         embs = [sentence_embeddings[ix] for ix in doc.detail_sentences_nums]
#         avg_embs.append(np.mean(embs, axis=0))
#
#     avg_embs = np.array(avg_embs)
#     print('avg_embs shape:', avg_embs.shape)
#
#     utils.save(avg_embs, 'bert_detail_embeddings_sentences_avg')
