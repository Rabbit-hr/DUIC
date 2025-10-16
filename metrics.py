import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import mutual_info_score

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    sum = 0.0
    list_index = list(zip(*ind))
    for i, j in list_index:
        sum = sum+ w[i,j]
    return sum * 1.0 / y_pred.size

def FN_TP_TN_FP(predict_lable,real_lable):
    pre = list(predict_lable)
    rea = list(real_lable)
    # tp+fp
    tp_fp = 0
    pre_lable_num = {}
    for tem in pre:
        if tem not in pre_lable_num:
            pre_lable_num[tem] = 1.0
        else:
            pre_lable_num[tem] += 1.0
    for tem in pre_lable_num.items():
        tp_fp += (tem[1]*(tem[1]-1))/2

    # tn+fn
    tn_fn = 0.0
    lll = list(pre_lable_num.items())
    for i in range(len(lll)-1):
        for j in range(i+1,len(lll)):
            tn_fn += lll[i][1]*lll[j][1]

    # tp
    tp = 0
    pre_lable_num.clear()
    for i in range(len(pre)):
        if pre[i] not in pre_lable_num:
            pre_lable_num[pre[i]] = str(rea[i])
        else:
            pre_lable_num[pre[i]] = pre_lable_num[pre[i]]+" "+str(rea[i])
    for tem in pre_lable_num.items():
        tem_dict = {}
        for temm in tem[1].split():
            if temm not in tem_dict:
                tem_dict[temm] = 1.0
            else:
                tem_dict[temm] += 1.0
        for temmm in tem_dict.items():
            if temmm[1] > 1:
                tp += (temmm[1]*(temmm[1]-1))/2
        tem_dict.clear()

    # fn
    fn = 0.0
    lll = list(pre_lable_num.items())
    for i in range(len(lll) - 1):
        for j in range(i + 1, len(lll)):
            cur_split = list(lll[i][1].split())
            aft_split = list(lll[j][1].split())
            temcuronly = []
            for tem in cur_split:
                if tem not in temcuronly:
                    temcuronly.append(tem)
            for tem in temcuronly:
                fn += cur_split.count(tem)*aft_split.count(tem)

    enddict = {}
    enddict["TP"] = tp
    enddict["FP"] = tp_fp-tp
    enddict["TN"] = tn_fn-fn
    enddict["FN"] = fn
    # print(enddict)
    return enddict

def Accuracy(predict_lable,real_lable):
    """
        Calculate accuracy.
    """
    fn_tp_tn_fp = FN_TP_TN_FP(predict_lable, real_lable)
    return (fn_tp_tn_fp["TP"]+fn_tp_tn_fp["TN"])/(fn_tp_tn_fp["TP"]+fn_tp_tn_fp["FN"]+fn_tp_tn_fp["FP"]+fn_tp_tn_fp["TN"])



def variation_of_information(y_true, y_pred):
    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log(probs + 1e-10))

    H_true = entropy(y_true)
    H_pred = entropy(y_pred)
    I = mutual_info_score(y_true, y_pred)
    VI = H_true + H_pred - 2 * I
    return VI