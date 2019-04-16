import pandas as pd
import os
import utils
import langid
import re
import numpy as np
import collections
from sklearn.preprocessing import LabelEncoder


class Doc:
    title_words = None
    title_words_nums = None

    detail_words = None
    detail_words_nums = None

    detail_sentences = None
    detail_sentences_nums = None
    detail_sentences_words_nums = None

    # no pre-processed original data
    def __init__(self, title, detail, target, tag, title_words, detail_words, detail_sentences):
        self.title = title
        self.detail = detail
        self.target = target
        self.tag = tag
        self.title_words = title_words
        self.detail_words = detail_words
        self.detail_sentences = detail_sentences


def normalize_text(text):
    # Lower case
    norm_text = text.lower()
    # Pad punctuation with spaces on both sides
    # Normal punctuations except \'
    norm_text = re.sub(r'([!\"#$%&\(\)\*\+,-\.\/:;<=>\?@\[\\\]\^_`\{\|\}~])', ' \\1 ', norm_text)
    norm_text = re.sub('\s{2,}', ' ', norm_text)
    return norm_text


def split_into_sentences(text):
    norm_text = normalize_text(text)
    sentences = [x.strip() for x in re.split(r"[\.\n]+", norm_text)]
    sentences = [x for x in sentences if len(x) > 0]
    return sentences


def csv_to_pkl(csv_file, pkl_file, cols, lan_chk):
    # Get params
    summary = cols[0]
    detail = cols[1]
    label = cols[2]

    # Read file
    df = pd.read_csv(csv_file)

    # Not null filter
    mask = (df[summary].notnull()) & (df[detail].notnull()) & (df[label].notnull())
    df = df.loc[mask]
    print('not null filtered dataset size:', len(df))

    # Language filter
    if lan_chk:
        df[summary + ' Lan'] = df[summary].apply(lambda x: langid.classify(x)[0])
        df[detail + ' Lan'] = df[detail].apply(lambda x: langid.classify(x)[0])
        mask = (df[summary + ' Lan'] == 'en') | (df[detail + ' Lan'] == 'en')
        df = df.loc[mask]
    print('language filtered dataset size:', len(df))

    # To pickle
    df.to_pickle(pkl_file)


def pkl_to_doc(pkl_file, cols):
    # Get params
    summary = cols[0]
    detail = cols[1]
    label = cols[2]

    # Read file
    df = pd.read_pickle(pkl_file)

    # Generate data
    titles = df[summary].values.tolist()
    details = df[detail].values.tolist()
    targets = df[label].values.tolist()
    assert len(titles) == len(details) == len(targets)

    titles_words = [normalize_text(x).split() for x in titles]
    details_words = [normalize_text(x).split() for x in details]
    details_sentences = [split_into_sentences(x) for x in details]

    titles = [' '.join(x) for x in titles_words]
    details = [' '.join(x) for x in details_words]

    # Label encoder
    label_encoder = LabelEncoder()
    targets = label_encoder.fit_transform(np.array(targets))

    # Create docs
    all_docs = []
    for i in range(len(titles)):
        all_docs.append(Doc(titles[i], details[i], targets[i], i,
                            titles_words[i], details_words[i], details_sentences[i]))
    print('all docs length:', len(all_docs))

    utils.save(all_docs, 'all_docs')


def init_common_dataset(dataset_csv_file, cols, lan_chk=True):
    csv = os.path.join('..', 'dataset', dataset_csv_file + '.csv')
    pkl = os.path.join('..', 'dataset', dataset_csv_file + '.pkl')
    print('csv to pkl...')
    csv_to_pkl(csv, pkl, cols, lan_chk)
    print('pkl to doc...')
    pkl_to_doc(pkl, cols)


def build_dictionary(docs, min_count, remove_top_num):
    # Get all the words
    title_words_list = [doc.title_words for doc in docs]
    words = [word for inner_list in title_words_list for word in inner_list]
    detail_words_list = [doc.detail_words for doc in docs]
    detail_words = [word for inner_list in detail_words_list for word in inner_list]
    words.extend(detail_words)

    # Initialize dict
    word_dict = {'<unk>': 0, '<pad>': 1}

    # Count words
    count = collections.Counter(words).most_common()
    count = [word for word, word_count in count if word_count >= min_count]

    # Remove top frequency words
    count = count[remove_top_num:]

    # Now create the dictionary
    for word in count:
        word_dict[word] = len(word_dict)

    word_dict_rev = dict(zip(word_dict.values(), word_dict.keys()))

    return word_dict, word_dict_rev


def add_numbers_to_docs(all_docs, word_dict):
    sentence_ix = 0
    for doc in all_docs:
        # title words numbers
        nums = []
        for word in doc.title_words:
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = 0
            nums.append(word_ix)
        doc.title_words_nums = nums

        # detail words numbers
        nums = []
        for word in doc.detail_words:
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = 0
            nums.append(word_ix)
        doc.detail_words_nums = nums

        # detail_sentences_words_nums_dbow
        sentences_words_nums = []
        for sentence in doc.detail_sentences:
            nums = []
            for word in sentence.split():
                if word in word_dict:
                    word_ix = word_dict[word]
                else:
                    word_ix = 0
                nums.append(word_ix)
            sentences_words_nums.append(nums)
        doc.detail_sentences_words_nums = sentences_words_nums

        # detail_sentences_nums
        doc.detail_sentences_nums = [x + sentence_ix for x in range(len(doc.detail_sentences))]
        sentence_ix += len(doc.detail_sentences)

    return all_docs


def pre_process(min_count=2, remove_top_num=0):
    print('Preprocessing dataset')

    # Load
    all_docs = utils.load('all_docs')
    print('all docs length:', len(all_docs))

    # Build dictionary
    print('Creating dictionary')
    word_dictionary, word_dictionary_rev = build_dictionary(all_docs, min_count, remove_top_num)
    vocabulary_size = len(word_dictionary)
    print('Vocabulary size:', vocabulary_size)

    # Text to numbers
    all_docs = add_numbers_to_docs(all_docs, word_dictionary)

    # Save
    utils.save(all_docs, 'all_docs')
    utils.save(word_dictionary, 'word_dictionary')
    utils.save(word_dictionary_rev, 'word_dictionary_rev')


def generate_batch_data(train_data, label_data, batch_size, generation):
    start = generation * batch_size
    end = (generation + 1) * batch_size
    batch_train_data = train_data[start: end]
    batch_label_data = label_data[start: end]

    if len(batch_train_data) < batch_size:
        rand_num = batch_size - len(batch_train_data)
        rand_ix = np.random.choice(len(train_data), size=rand_num)
        rand_batch = np.array([train_data[x] for x in rand_ix])
        rand_label = np.array([label_data[x] for x in rand_ix])
        batch_train_data = np.append(batch_train_data, rand_batch, axis=0)
        batch_label_data = np.append(batch_label_data, rand_label, axis=0)

    return batch_train_data, batch_label_data
