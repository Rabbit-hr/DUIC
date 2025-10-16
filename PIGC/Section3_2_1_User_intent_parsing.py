import pickle as pkl
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm

def pkl_save(root, info):
    with open(root, 'wb') as f:
        pkl.dump(info, f)

def pkl_read(root):
    with open(root, 'rb') as f:
        return pkl.load(f)

def tf_idf_toarray(data):
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(data)
    return tfidf_matrix.toarray()

def re_l2(u):
    u = np.array(u)
    norm_val = np.sqrt(np.sum(np.square(u), axis=1, keepdims=True))
    norm_val[norm_val == 0] = 1
    return u / norm_val

def creat_term_vec(term_list):
    model_path = 'English_wiki/wiki.en.vec'
    word_vectors = KeyedVectors.load_word2vec_format(model_path, limit=30000)
    vectors = np.array([word_vectors[word] for word in term_list if word in word_vectors])
    return vectors

def creat_tfidf_array(x):
    one_zero = np.where(x > 0, 1, 0)
    x_tfidf = tf_idf_toarray(x)
    x_tfidf = x_tfidf * one_zero
    norm_val = np.sqrt(np.sum(np.square(x_tfidf), axis=1, keepdims=True)) + 1e-8
    return x_tfidf / norm_val

def compute_intent_mask(term_vec, query_vec):
    """ Eq.(2) return: mask ∈ {0,1}^V """
    sim_matrix = cosine_similarity(term_vec, query_vec)  # V×K
    Mv = np.mean(sim_matrix, axis=1)
    mu_rel = np.mean(Mv)
    sigma_rel = np.std(Mv)
    tau_M = mu_rel + sigma_rel
    mask = (Mv > tau_M).astype(float)
    return mask


def creat_doc_vec(x, term_vec, mask=None):
    """[(A_text ⊙ M) ⋅ W] Eq.(1)"""
    x_tfidf = creat_tfidf_array(x)
    if mask is not None:
        x_tfidf = x_tfidf * mask  # 应用mask

    doc_vecs = []
    for i, row in enumerate(x_tfidf):
        nonzero_idx = np.nonzero(row)[0]
        if len(nonzero_idx) == 0:
            doc_vecs.append(np.zeros(term_vec.shape[1]))
            continue
        weighted_terms = [row[j] * term_vec[j] for j in nonzero_idx]
        doc_vec = np.mean(weighted_terms, axis=0)
        doc_vecs.append(doc_vec)
    return np.array(doc_vecs)

def calculate_intent_alignment_score(query_list, x, term_vec, score_save_path):
    """ Eq.(1)"""
    query_vec = creat_term_vec(query_list)
    # mask 𝕄（Eq.2）
    mask = compute_intent_mask(term_vec, query_vec)
    doc_vec = creat_doc_vec(x, term_vec, mask)
    # Intent alignment score S
    similarity_matrix = cosine_similarity(doc_vec, query_vec)

    mean = np.mean(similarity_matrix, axis=1, keepdims=True)
    std = np.std(similarity_matrix, axis=1, keepdims=True) + 1e-6
    i_score = (similarity_matrix - mean) / std
    i_score = np.maximum(i_score, 0)

    pkl_save(score_save_path, i_score)
    return i_score


def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-8)