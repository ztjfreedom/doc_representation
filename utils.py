import os
import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA


def cosine_dist(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def exists(file_name):
    return os.path.exists(os.path.join('temp', file_name + '.pkl'))


def save(save_obj, file_name):
    print('Saving', file_name)
    with open(os.path.join('temp', file_name + '.pkl'), 'wb') as f:
        pickle.dump(save_obj, f)


def load(file_name):
    with open(os.path.join('temp', file_name + '.pkl'), 'rb') as f:
        obj = pickle.load(f)
    return obj


def save_doc_embeddings(save_obj, model_name, is_concat=False, window_size=None, batch_size=None, negative_size=None):
    file_name = model_name
    if is_concat:
        file_name += '_concat'
    if window_size is not None:
        file_name += '_' + str(window_size)
    if batch_size is not None:
        file_name += '_' + str(batch_size)
    if negative_size is not None:
        file_name += '_' + str(negative_size)
    file_name += '.pkl'

    print('Saving file name:', file_name)
    with open(os.path.join('doc_embeddings', file_name), 'wb') as f:
        pickle.dump(save_obj, f)


def load_doc_embeddings(file):
    with open(os.path.join('doc_embeddings', file + '.pkl'), 'rb') as f:
        doc_emb = pickle.load(f)
    return doc_emb


def load_all_doc_embeddings():
    doc_emb_dict = {}
    for file_name in os.listdir('doc_embeddings'):
        path = os.path.join('doc_embeddings', file_name)
        if not os.path.isdir(path):
            with open(path, 'rb') as f:
                key = file_name[: file_name.rfind('.')]
                val = pickle.load(f)
                doc_emb_dict[key] = val
    return doc_emb_dict


def split_embeddings(embeddings, split_num):
    corpus_size = len(embeddings) // split_num
    emb_list = []
    for i in range(split_num):
        emb = []
        for j in range(corpus_size):
            emb.append(embeddings[i * corpus_size + j])
        emb_list.append(emb)

    emb_list = np.array(emb_list)
    print('embedding size after split:', emb_list.shape)
    return np.array(emb_list)


def concat_embeddings(embeddings_list):
    emb_len = len(embeddings_list[0])
    for emb in embeddings_list:
        assert len(emb) == emb_len

    concat_emb = np.concatenate(embeddings_list, axis=1)
    print('embeddings shape after concat:', concat_emb.shape)

    return concat_emb


def mean_embeddings(embeddings_list):
    emb_len = len(embeddings_list[0])
    for emb in embeddings_list:
        assert len(emb) == emb_len

    avg_emb = np.mean(embeddings_list, axis=0)
    print('embeddings shape after mean:', avg_emb.shape)

    return avg_emb


def normalize_embeddings(embeddings):
    return preprocessing.scale(embeddings)


def reduce_dim(vecs, target_dim):
    pca = PCA(n_components=target_dim)
    vecs = pca.fit_transform(vecs)
    print('shape after PCA:', vecs.shape)
    return vecs


if __name__ == '__main__':
    X = [1, 2]
    num = 1
    print(split_embeddings(X, 2))

