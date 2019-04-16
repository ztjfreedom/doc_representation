import bert


def show_length_info():
    bert.show_all_length_info()


def word_emb():
    sample_words = ['engineer', 'android', 'collect']
    bert.create_word_emb()
    bert.test_word_emb(sample_words)


def summary_emb():
    bert.create_summary_emb()


def detail_emb():
    bert.create_detail_emb()


if __name__ == '__main__':
    show_length_info()
    # word_emb()
    # summary_emb()
    detail_emb()
