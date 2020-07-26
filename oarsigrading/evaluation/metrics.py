from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import itertools
from sklearn.metrics import confusion_matrix


def compute_metrics(val_gt, val_pred, no_kl=False):
    kappas = []
    acc = []
    device = val_gt.device
    val_gt = val_gt.squeeze().to('cpu').numpy()
    val_pred = val_pred.squeeze().to('cpu').numpy()

    for task_id in range(val_gt.shape[1]):
        kappas.append(cohen_kappa_score(val_gt[:, task_id], val_pred[:, task_id], weights='quadratic'))
        acc.append(balanced_accuracy_score(val_gt[:, task_id], val_pred[:, task_id]))

    res = {}
    ind = 0  # This variable is needed to handle the case when kl grade is not used
    for feature in ['kl', 'ostl', 'osfl', 'jsl', 'ostm', 'osfm', 'jsm']:
        if no_kl and feature == 'kl':
            continue
        value = kappas[ind]
        if np.isnan(value):
            value = 0.
        res[f"kappa_{feature}"] = torch.tensor(value).to(device)
        value = acc[ind]
        if np.isnan(value):
            value = 0.
        res[f"acc_{feature}"] = torch.tensor(value).to(device)
        ind += 1

    return res


def bootstrap_ci(metric, y, preds, n_bootstrap, seed=12345, stratified=True, alpha=95):
    """
    Parameters
    ----------
    metric : fucntion
        Metric to compute, e.g. AUC for ROC curve or AP for PR curve
    y : numpy.array
        Ground truth
    preds : numpy.array
        Predictions
    n_bootstrap:
        Number of bootstrap samples to draw
    seed : int
        Random seed
    stratified : bool
        Whether to do a stratified bootstrapping
    alpha : float
        Confidence intervals width
    """

    np.random.seed(seed)
    metric_vals = []
    classes = np.unique(y)
    inds = []
    for cls in classes:
        inds.append(np.where(y == cls)[0])

    for _ in tqdm(range(n_bootstrap), total=n_bootstrap, desc='Bootstrap:'):
        if stratified:
            ind_bs = []
            for ind_cur in inds:
                ind_bs.append(np.random.choice(ind_cur, ind_cur.shape[0]))
            ind = np.hstack(ind_bs)
        else:
            ind = np.random.choice(y.shape[0], y.shape[0])

        if y[ind].sum() == 0:
            continue
        metric_vals.append(metric(y[ind], preds[ind]))

    metric_val = metric(y, preds)
    ci_l = np.percentile(metric_vals, (100 - alpha) // 2)
    ci_h = np.percentile(metric_vals, alpha + (100 - alpha) // 2)

    return metric_val, ci_l, ci_h


def plot_confusion(y_gt, y_pred, save_name, show=False, font=16):
    cm = confusion_matrix(y_gt, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    matplotlib.rcParams.update({'font.size': font})
    plt.figure(figsize=(6, 6))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens, resample=False)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(np.round(cm[i, j] * 100, 2)),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.xticks(np.arange(cm.shape[0], dtype=int), np.arange(cm.shape[0], dtype=int))
    plt.yticks(np.arange(cm.shape[1], dtype=int), np.arange(cm.shape[1], dtype=int))
    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(save_name.replace(' ', '_'), bbox_inches='tight')
    if not show:
        plt.close()
    else:
        plt.show()
