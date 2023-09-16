# -*- coding: utf-8 -*-

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import classification_report


def ROC(model, testx, testy, s=''):
    #    acc = model.evaluate(testx,testy)[1]
    prey = model.predict(testx, verbose=1)
    testy = testy

    fpr, tpr, thresholds = metrics.roc_curve(testy, prey)
    auc = metrics.roc_auc_score(testy, prey)
    print("AUC : ", auc)
    plt.figure()
    # plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0, 1], [0, 1], color='navy', lw='2', linestyle='--', label='Random')
    plt.plot(fpr, tpr, label='ROC curve (AUC = {:.3f})'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('auROC Curve of ' + s)
    plt.legend(loc='best')
    plt.savefig("../Result/confusion_train/234/roc/" + s + "auROC.png")
    plt.close('all')


def show_train_history(train_history, s='', locate=''):
    plt.plot(list(range(1, len(train_history['accuracy']) + 1)), train_history['accuracy'], color='red')
    plt.plot(list(range(1, len(train_history['val_accuracy']) + 1)), train_history['val_accuracy'], color='blue')

    plt.title('Accuracy under ' + s)
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.grid()
    plt.savefig(locate + 'Acc_' + s + '.png')
    plt.close('all')

    plt.plot(list(range(1, len(train_history['loss']) + 1)), train_history['loss'], color='red')
    plt.plot(list(range(1, len(train_history['val_loss']) + 1)), train_history['val_loss'], color='blue')

    plt.title('Loss under ' + s)
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.grid()
    plt.savefig(locate + 'Loss_' + s + '.png')
    plt.close('all')

def draw(list1):
    sum=len(list1)
    plt.figure(figsize=(24,16))
    plt.plot(list1[0], [i*1.1+5 for  i in list1[1]], label='express', c='y')
    plt.plot(list1[0], [i*0.9-5 for  i in list1[1]], label='express', c='y')
    plt.scatter(list1[0],list1[2],label='predict',s=1,c='b')
    plt.plot(list1[0],list1[1],label='express',c='r')
    plt.title('0~500')
    plt.ylabel('express')
    plt.xlabel('number')
    plt.legend(loc='best')
    plt.savefig('../Result/exp0~10.png')
    plt.close('all')