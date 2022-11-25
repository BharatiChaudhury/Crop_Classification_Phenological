import read_rsdata
import numpy as np
import matplotlib.pyplot as plt
d1 = read_rsdata.prepare_input_data(None, data = 'hx', conv = '2D', split = 'kfold')
d2 = read_rsdata.prepare_input_data(None, data = 'mx', conv = '2D', split = 'kfold')

hx,cg,ct = d1()
mx,_,_ = d2()

datal = cg
crop = ct
datam = mx
datah = hx

pr_ct = results[0] ##Prediction results of crop type
pr_cg = results[1] ##Prediction results of crop growth stage

gt_cg = np.argmax(pr_cg, axis=1)
gt_ct = np.argmax(pr_ct, axis=1)

ct_pred = np.argmax(ct_test, axis=1)
cg_pred = np.argmax(cg_test, axis=1)
cg_all_pred, cg_all = [], []
for i in range(len(ct_pred)):
    if(ct_pred[i] == 0 and cg_pred[i] == 0):
        cg_all_pred.append(1)
        if(gt_cg[i]==0):
            cg_all.append(1)
        else:
            cg_all.append(0)
            
kappa_coefficient_ct = cohen_kappa_score(cg_all_pred, cg_all)
print('kappa_coefficient_ct: {}'.format(kappa_coefficient_ct))
precision, recall, fscore, support = score(cg_all_pred, cg_all)
print('cg{}'.format(classification_report(cg_all_pred, cg_all)))
fusion
cg_all_pred, cg_all = [], []
for i in range(len(ct_pred)):
    if(ct_pred[i] == 0 and cg_pred[i] == 1):
        cg_all_pred.append(1)
        if(gt_cg[i]==1):
            cg_all.append(1)
        else:
            cg_all.append(0)
            
kappa_coefficient_ct = cohen_kappa_score(cg_all_pred, cg_all)
print('kappa_coefficient_ct: {}'.format(kappa_coefficient_ct))
precision, recall, fscore, support = score(cg_all_pred, cg_all)
print('cg{}'.format(classification_report(cg_all_pred, cg_all)))
cg_all_pred, cg_all = [], []
for i in range(len(ct_pred)):
    if(ct_pred[i] == 0 and cg_pred[i] == 2):
        cg_all_pred.append(1)
        if(gt_cg[i]==2):
            cg_all.append(1)
        else:
            cg_all.append(0)
            
kappa_coefficient_ct = cohen_kappa_score(cg_all_pred, cg_all)
print('kappa_coefficient_ct: {}'.format(kappa_coefficient_ct))
precision, recall, fscore, support = score(cg_all_pred, cg_all)
print('cg{}'.format(classification_report(cg_all_pred, cg_all)))
cg_all_pred, cg_all = [], []
for i in range(len(ct_pred)):
    if(ct_pred[i] == 1 and cg_pred[i] == 0):
        cg_all_pred.append(1)
        if(gt_cg[i]==0):
            cg_all.append(1)
        else:
            cg_all.append(0)
            
kappa_coefficient_ct = cohen_kappa_score(cg_all_pred, cg_all)
print('kappa_coefficient_ct: {}'.format(kappa_coefficient_ct))
precision, recall, fscore, support = score(cg_all_pred, cg_all)
print('cg{}'.format(classification_report(cg_all_pred, cg_all)))
            

