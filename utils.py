##TODO -> PCA on Hx
##TODO -> Images as patch for Hx
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay,classification_report

def PCA_(X):
    numComponents = 30
    pca = PCA(n_components = numComponents)
    newX = np.reshape(X, (-1, X.shape[2]))
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX

def plot_multi_task_loss(model_history):

    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('multi_task_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(model_history.history['ct_loss'])
    plt.plot(model_history.history['val_ct_loss'])
    plt.title('crop_type')
    plt.ylabel('crop_type_loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(model_history.history['cg_loss'])
    plt.plot(model_history.history['val_cg_loss'])
    plt.title('crop_growth_stage')
    plt.ylabel('crop_growth_loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
# summarize history for loss
def plot_single_task_loss(model_history,label):
    
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title(label)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def precision_recall_f1score(ct_test,cg_test,results):
    ## Dilation factors = [1,2,4,8,16,32]

    print("Precision,Recall, F1-Score")
    
    pr_ct = results[0] ##Prediction results of crop type
    pr_cg = results[1] ##Prediction results of crop growth stage

    gt_cg = np.argmax(pr_cg, axis=1)
    gt_ct = np.argmax(pr_ct, axis=1)

    ct_pred = np.argmax(ct_test, axis=1)
    cg_pred = np.argmax(cg_test, axis=1)


    kappa_coefficient_ct = cohen_kappa_score(gt_ct, ct_pred)
    print('kappa_coefficient_ct: {}'.format(kappa_coefficient_ct))

    precision, recall, fscore, support = score(gt_cg, cg_pred)
    kappa_coefficient_cg = cohen_kappa_score(gt_cg, cg_pred)

    print('kappa_coefficient_cg: {}'.format(kappa_coefficient_cg))
    
    print('cg{}'.format(classification_report(gt_cg, cg_pred)))
    print('ct{}'.format(classification_report(gt_ct,ct_pred)))
    matrix_ct = confusion_matrix(gt_ct, ct_pred)
    print(matrix_ct.diagonal()/matrix_ct.sum(axis=1))

    matrix_cg = confusion_matrix(gt_cg, cg_pred)
    print(matrix_cg.diagonal()/matrix_cg.sum(axis=1))


    labels = ["Corn", "Cotton", "Soybean"]
    disp_ct = ConfusionMatrixDisplay(confusion_matrix=matrix_ct, display_labels=labels)

    disp_ct.plot()

    labels = ["Early", "Mid", "Harvest"]
    disp_cg = ConfusionMatrixDisplay(confusion_matrix=matrix_cg, display_labels=labels)

    disp_cg.plot()
    plt.show()



def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX
    
def createImageCubes(X, windowSize=5):
    for i in range(len(X)):
        margin = int((windowSize - 1) / 2)
        margin = 2
        zeroPaddedX = padWithZeros(X, margin=margin)
        patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
        patchIndex = 0
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
                patchesData[patchIndex, :, :, :] = patch
                patchIndex = patchIndex + 1
        #patches_gt.append(label1)
        #patches_gt.append(label2)
    return patchesData

def check_crop_growth_stage_precision_recall_f1score(results, ct_test,cg_test):

    pr_ct = results[0] ##Prediction results of crop type
    pr_cg = results[1] ##Prediction results of crop growth stage

    gt_cg = np.argmax(pr_cg, axis=1)
    gt_ct = np.argmax(pr_ct, axis=1)

    ct_pred = np.argmax(ct_test, axis=1)
    cg_pred = np.argmax(cg_test, axis=1)
    cg_all_pred, cg_all = [], []
    for i in range(len(ct_pred)):
        if(ct_pred[i] == 0 and cg_pred[i] == 0): ##crop growth stage change from 0 to 2
            cg_all_pred.append(1)
        if(gt_cg[i]==0): ##predicted crop type and respective growth stage
            cg_all.append(1)
        else:
            cg_all.append(0)
            
    kappa_coefficient_ct = cohen_kappa_score(cg_all_pred, cg_all)
    print('kappa_coefficient_ct: {}'.format(kappa_coefficient_ct))
    precision, recall, fscore, support = score(cg_all_pred, cg_all)
    print('cg{}'.format(classification_report(cg_all_pred, cg_all)))