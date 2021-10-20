import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

print ("CIFAR100 vs SVHN")
nll_svhn = np.load('svhn100_nll_vae.npy')
nll_cifar = np.load('cifar100_nll_vae.npy')
combined = np.concatenate((nll_cifar, nll_svhn))
label_1 = np.ones(len(nll_cifar))
label_2 = np.zeros(len(nll_svhn))
label = np.concatenate((label_1, label_2))
fpr, tpr, thresholds = metrics.roc_curve(label, combined, pos_label=0)
# plot_roc_curve(fpr, tpr)
rocauc = metrics.auc(fpr, tpr)
aucprc = metrics.average_precision_score(label, -combined)
fpr80 = (fpr[np.argmin(np.abs(tpr - 0.8))].min())

print ("AUROC:", rocauc, "AUPR:",aucprc, "FPR80:",fpr80)

print ("CIFAR100 vs CelebA")
nll_svhn = np.load('cel100_nll_vae.npy')
nll_cifar = np.load('cifar100_nll_vae.npy')
combined = np.concatenate((nll_cifar, nll_svhn))
label_1 = np.ones(len(nll_cifar))
label_2 = np.zeros(len(nll_svhn))
label = np.concatenate((label_1, label_2))
fpr, tpr, thresholds = metrics.roc_curve(label, combined, pos_label=0)
# plot_roc_curve(fpr, tpr)
rocauc = metrics.auc(fpr, tpr)
aucprc = metrics.average_precision_score(label, -combined)
fpr80 = (fpr[np.argmin(np.abs(tpr - 0.8))].min())

print ("AUROC:", rocauc, "AUPR:",aucprc, "FPR80:",fpr80)