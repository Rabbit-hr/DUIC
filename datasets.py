import pickle as pkl
import numpy as np
import scanpy as sc

#save information
def pkl_save(root,info):
    ff = open(root, 'wb')
    pkl.dump(info, ff)
    ff.close()
#read pickle informat
def pkl_read(root):
    with open(root, "rb") as f:
        info = pkl.load(f)
    return info

def load_20NG3():
    x = pkl_read('data/20NG3/cbow.pkl')
    term_list = pkl_read('data/20NG3/term_list.pkl')
    y = pkl_read('data/20NG3/labels.pkl')
    terms_vectors = pkl_read('data/' + '20NG3' + '/terms_vectors.pkl')
    return x, y, term_list,terms_vectors

def load_BBCSport():
    adata = sc.read('data/BBCSport/bbcsport.mtx')
    data = adata.X
    data = data.todense()
    data = data.A
    data = np.transpose(data)
    labels = []
    with open('data/BBCSport/bbcsport.classes') as file2:
        labels_pre = file2.readlines()[4:]
    for each in labels_pre:
        each = each.strip("\n").split(" ")
        labels.append(int(each[1]))
    labels = np.array(labels)
    terms_vectors = pkl_read('data/' + 'BBCSport' + '/terms_vectors.pkl')

    with open('data/BBCSport/bbcsport.terms', 'r') as f_w:
        content = f_w.read()
        word_list = content.split('\n')
        f_w.close()
    return data,labels,word_list,terms_vectors


def load_data(dataset_name):
    if dataset_name == '20NG3':
        return load_20NG3()
    elif dataset_name == 'BBCSport':
        return load_BBCSport()
    else:
        print('Not defined for loading', dataset_name)
        exit(0)