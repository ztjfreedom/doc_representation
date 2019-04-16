import bert
import utils


def show_length_info():
    word_dictionary = utils.load('word_dictionary')
    bert.show_words_length_info(word_dictionary)

    # for i in [3, 6, 9, 12]:
    #     seq_name = 'window_seq_' + str(i)
    #     window_seq_list = utils.load(seq_name)
    #     seq_len = [len(seq) for seq in window_seq_list]
    #     print(seq_name + ':')
    #     print('list size:', len(window_seq_list))
    #     print('max length of seq:', max(seq_len))
    #     print('avg length of seq:', sum(seq_len) / len(window_seq_list))

    all_docs = utils.load('all_docs')
    texts = [doc.texts for doc in all_docs]
    texts_len = [len(text) for text in texts]
    print('max length of texts:', max(texts_len))
    print('avg length of texts:', sum(texts_len) / len(texts))


def word_emb():
    bert.create_word_emb()


def window_seq_emb(window_size):
    seq_name = 'window_seq_' + str(window_size)
    save_file_name = 'bert_' + seq_name + '_embeddings'
    window_seq_list = utils.load(seq_name)
    bert.create_emb(window_seq_list, save_file_name)


def doc_emb():
    all_docs = utils.load('all_docs')
    texts = [doc.texts for doc in all_docs]
    bert.create_emb(texts, 'bert_doc_embeddings')


if __name__ == '__main__':
    show_length_info()
    # word_emb()
    doc_emb()
    # window_seq_emb(3)
    # window_seq_emb(6)
    # window_seq_emb(9)
    # window_seq_emb(12)
