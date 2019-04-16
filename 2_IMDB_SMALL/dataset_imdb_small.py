import utils
import dataset_imdb
import string
import re
from dataset_imdb import Doc


def pre_process(min_count=2, remove_top_num=0, sub_size=100):
    print('Preprocessing dataset small')

    # Load
    all_docs, _, _ = dataset_imdb.load_dataset()
    all_docs_small = []
    for i in range(8):
        sub_docs = all_docs[12500*i: 12500*i+sub_size]
        all_docs_small.extend(sub_docs)

    # Modify tag
    for i in range(len(all_docs_small)):
        all_docs_small[i].tag = i

    # Process texts
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    texts = [' '.join(doc.words) for doc in all_docs_small]
    texts = [regex.sub('', text) for text in texts]
    texts = [re.sub('\s{2,}', ' ', text).strip() for text in texts]
    assert len(all_docs_small) == len(texts)

    for i in range(len(all_docs_small)):
        all_docs_small[i].texts = texts[i]
        all_docs_small[i].words = texts[i].split(' ')

    train_docs_small = [doc for doc in all_docs_small if doc.split == 'train']
    test_docs_small = [doc for doc in all_docs_small if doc.split == 'test']

    print('%d docs: %d train-sentiment, %d test-sentiment' %
          (len(all_docs_small), len(train_docs_small), len(test_docs_small)))

    # Build dictionary
    print('Creating dictionary')
    word_dictionary_small, word_dictionary_rev_small = dataset_imdb.build_dictionary(all_docs_small,
                                                                                     min_count, remove_top_num)
    vocabulary_size = len(word_dictionary_small)
    print('Vocabulary size:', vocabulary_size)

    # Text to numbers
    all_docs_small = dataset_imdb.add_numbers_to_docs(all_docs_small, word_dictionary_small)

    # Save
    utils.save(all_docs_small, 'all_docs')
    utils.save(train_docs_small, 'train_docs')
    utils.save(test_docs_small, 'test_docs')
    utils.save(word_dictionary_small, 'word_dictionary')
    utils.save(word_dictionary_rev_small, 'word_dictionary_rev')

    for i in [3, 6, 9, 12]:
        window_seq_list = create_window_seq(i)
        utils.save(window_seq_list, 'window_seq_' + str(i))


def create_window_seq(window_size):
    window_seq_list = []
    all_docs = utils.load('all_docs')
    for doc in all_docs:
        if len(doc.words) <= window_size:
            padded = ['.'] * (window_size + 1 - len(doc.words))
            padded.extend(doc.words)
            doc.words = padded
            window_seq_list.append(' '.join(doc.words))
        else:
            window_seq_list.extend(' '.join(doc.words[i: i + window_size])
                                   for i in range(0, len(doc.words) - window_size))

    return window_seq_list
