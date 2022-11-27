import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dgl
import torch as torch
from scipy import interp
from sklearn import metrics
import warnings,sys
import networkx as nx
import prettytable as pt
from train import Train_loop


if __name__ == '__main__':
    #dgl.backend.load_backend('pytorch')
    warnings.filterwarnings("ignore")
    #dgl.load_backend('pytorch')
    device = torch.device("cpu")
    #device = torch.device("cuda:1")
    
    lp_auc, lp_acc, lp_pre, lp_recall, lp_f1, lp_fprs, lp_tprs, \
    c_auc, c_hamming_loss, c_pre, c_recall, c_f1, c_fprs, c_tprs = Train_loop(epochs=30,
                                                                hidden_size=2048,
                                                                dropout=0.2,
                                                                slope=0.1,  # LeakyReLU
                                                                lr=0.0015,
                                                                wd=1e-3,
                                                                random_seed=42,
                                                                device=device)
   
    print('-AUC LP mean: %.4f, variance: %.4f \n' % (np.mean(lp_auc), np.std(lp_auc)),
          'Accuracy LP mean: %.4f, variance: %.4f \n' % (np.mean(lp_acc), np.std(lp_acc)),
          'Precision LP mean: %.4f, variance: %.4f \n' % (np.mean(lp_pre), np.std(lp_pre)),
          'Recall LP mean: %.4f, variance: %.4f \n' % (np.mean(lp_recall), np.std(lp_recall)),
          'F1-score LP mean: %.4f, variance: %.4f \n' % (np.mean(lp_f1), np.std(lp_f1)),
          '-AUC C mean: %.4f, variance: %.4f \n' % (np.mean(c_auc), np.std(c_auc)),
          'Hamming loss C mean: %.4f, variance: %.4f \n' % (np.mean(c_hamming_loss), np.std(c_hamming_loss)),
          'Precision C mean: %.4f, variance: %.4f \n' % (np.mean(c_pre), np.std(c_pre)),
          'Recall C mean: %.4f, variance: %.4f \n' % (np.mean(c_recall), np.std(c_recall)),
          'F1-score C mean: %.4f, variance: %.4f \n' % (np.mean(c_f1), np.std(c_f1))
          )

    '''
    lp_auc_pd = pd.DataFrame(lp_auc)
    lp_fprs_pd = pd.DataFrame(lp_fprs)
    lp_tprs_pd = pd.DataFrame(lp_tprs)
    c_auc_pd = pd.DataFrame(c_auc)
    c_fprs_pd = pd.DataFrame(c_fprs)
    c_tprs_pd = pd.DataFrame(c_tprs)
    
    lp_auc_pd.to_csv('./data/debug_data/g_lp_auc.csv')
    lp_fprs_pd.to_csv('./data/debug_data/g_lp_fprs.csv')
    lp_tprs_pd .to_csv('./data/debug_data/g_lp_tprs.csv')
    c_auc_pd.to_csv('./data/debug_data/g_c_auc.csv')
    c_fprs_pd.to_csv('./data/debug_data/g_c_fprs.csv')
    c_tprs_pd .to_csv('./data/debug_data/g_c_tprs.csv')'''


    lp_mean_fpr = np.linspace(0, 1, 10000)
    lp_tpr = []

    for i in range(len(lp_fprs)):
        lp_tpr.append(interp(lp_mean_fpr, lp_fprs[i], lp_tprs[i]))
        lp_tpr[-1][0] = 0.0
        plt.plot(lp_fprs[i], lp_tprs[i], alpha=0.4, label='ROC fold %d (AUC = %.4f)' % (i + 1, lp_auc[i]))

    lp_mean_tpr = np.mean(lp_tpr, axis=0)
    lp_mean_tpr[-1] = 1.0
    lp_mean_auc = metrics.auc(lp_mean_fpr, lp_mean_tpr)
    lp_auc_std = np.std(lp_auc)
    plt.plot(lp_mean_fpr, lp_mean_tpr, color='b', alpha=0.8, label='Mean AUC (AUC = %.4f $\pm$ %.4f)' % (lp_mean_auc, lp_auc_std))

    lp_std_tpr = np.std(lp_tpr, axis=0)
    lp_tpr_upper = np.minimum(lp_mean_tpr + lp_std_tpr, 1)
    lp_tpr_lower = np.maximum(lp_mean_tpr - lp_std_tpr, 0)
    plt.fill_between(lp_mean_fpr, lp_tpr_lower, lp_tpr_upper, color='grey', alpha=0.3, label='$\pm$ 1 std.dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Link Prediction ROC Curves')
    plt.legend(loc='lower right')
    plt.show()

    # auc of c
    c_mean_fpr = np.linspace(0, 1, 10000)
    c_tpr = []

    for i in range(len(c_fprs)):
        c_tpr.append(interp(c_mean_fpr, c_fprs[i], c_tprs[i]))
        c_tpr[-1][0] = 0.0
        plt.plot(c_fprs[i], c_tprs[i], alpha=0.4, label='ROC fold %d (AUC = %.4f)' % (i + 1, c_auc[i]))

    c_mean_tpr = np.mean(c_tpr, axis=0)
    c_mean_tpr[-1] = 1.0
    c_mean_auc = metrics.auc(c_mean_fpr, c_mean_tpr)
    c_auc_std = np.std(c_auc)
    plt.plot(c_mean_fpr, c_mean_tpr, color='b', alpha=0.8, label='Mean AUC (AUC = %.4f $\pm$ %.4f)' % (c_mean_auc, c_auc_std))

    c_std_tpr = np.std(c_tpr, axis=0)
    c_tpr_upper = np.minimum(c_mean_tpr + c_std_tpr, 1)
    c_tpr_lower = np.maximum(c_mean_tpr - c_std_tpr, 0)
    plt.fill_between(c_mean_fpr, c_tpr_lower, c_tpr_upper, color='grey', alpha=0.3, label='$\pm$ 1 std.dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Classification ROC Curves')
    plt.legend(loc='lower right')
    plt.show()

    