import read_rsdata
from sklearn.model_selection import StratifiedKFold
import numpy as np
import tcn_
import tensorflow as tf 
import argparse
import gc
from keras.utils import np_utils
import model
import utils
def multi_task(x_train,ct_train,cg_train,cg_test,ct_test,x_test):
    loss_list = {'cg':'categorical_crossentropy','ct':'categorical_crossentropy'}
    metric_list = {'ct':'accuracy','cg':'accuracy'}
    loss_weights = {'cg':0.3,'ct':0.7}
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=20)
    neural_network.compile(loss = loss_list,optimizer = opt, metrics = metric_list, loss_weights=loss_weights)
    if fusion:
        history_cnn = neural_network.fit(x = [x_train[0],x_train[1]], y = [ct_train,cg_train], batch_size=25,validation_data=([x_test[0],x_test[1]],[ct_test,cg_test]),callbacks=[es],epochs=50,verbose=0)
        print(neural_network.evaluate([x_train[0],x_train[1]], [ct_train,cg_train], batch_size=20,verbose=0))
        scores = neural_network.evaluate([x_test[0],x_test[1]], [ct_test,cg_test], batch_size=20,verbose=0)
        utils.plot_multi_task_loss(history_cnn)

    else:
        history_cnn = neural_network.fit(x = x_train, y = [ct_train,cg_train], batch_size=25,epochs=50,verbose=0)
        print(neural_network.evaluate(x_train, [ct_train,cg_train], batch_size=20,verbose=0))
        scores = neural_network.evaluate(x_test, [ct_test,cg_test], batch_size=20,verbose=0)
        
    return scores

def singletask(x_train,y_train,y_test,x_test):
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    neural_network.compile(loss = 'categorical_crossentropy',optimizer = opt, metrics = ['accuracy'])
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=20)

    if fusion:
        history_cnn = neural_network.fit(x = [x_train[0],x_train[1]], y = y_train, validation_data = ([x_test[0],x_test[1]],y_test),callbacks=[es], batch_size=25,epochs=50,verbose=0)
        print(neural_network.evaluate([x_train[0],x_train[1]], y_train, batch_size=20,verbose=0))
        scores = neural_network.evaluate([x_test[0],x_test[1]], y_test, batch_size=20,verbose=0)
        utils.plot_single_task_loss(history_cnn,args.label)

        return scores
    else:
        history_cnn = neural_network.fit(x = x_train, y = y_train, validation_data = (x_test,y_test),callbacks=[es],batch_size=25,epochs=50,verbose=0)
        print(neural_network.evaluate(x_train, y_train, batch_size=20,verbose=0))
        scores = neural_network.evaluate(x_test, y_test, batch_size=20,verbose=0)
        return scores
    
def multitask_results(acc1_per_fold,loss1_per_fold,acc2_per_fold,loss2_per_fold): 
    print('........')
    print('Score per fold')
    for i in range(0, len(acc1_per_fold)):
        print('.......')
        print(f'>fold {i+1} - loss{loss1_per_fold[i]}- Accuracy:{acc1_per_fold[i]}%')
        print(f'>fold {i+1} - loss{loss2_per_fold[i]}- Accuracy:{acc2_per_fold[i]}%')
    
    print('.......')
    print('Average scores for all folds:')
    print(f'>Crop Type Accuracy:{np.mean(acc1_per_fold)}(+-{np.std(acc1_per_fold)})')
    print(f'>Crop Growth Stage Accuracy:{np.mean(acc2_per_fold)}(+-{np.std(acc2_per_fold)})')


    print(f'>Loss:{np.mean(loss1_per_fold)}')
    print(f'>Loss:{np.mean(loss2_per_fold)}')
      
    print('.......')   

def singletask_results(acc1_per_fold,loss1_per_fold,label):
    print('........')
    print('Score per fold')
    for i in range(0, len(acc1_per_fold)):
        print('.......')
        print(f'>fold {i+1} - loss{loss1_per_fold[i]}- Accuracy:{acc1_per_fold[i]}%')
        
    
    print('.......')
    print('Average scores for all folds:')
    print(label+ f'Accuracy:{np.mean(acc1_per_fold)}(+-{np.std(acc1_per_fold)})')

    print(label+f'Loss:{np.mean(loss1_per_fold)}')      
    print('.......')   

parser = argparse.ArgumentParser(description='Running Conv3D_2D_TCN')
parser.add_argument('--multitask', dest='multitask',type=int, help ='if multitask 1 else 0',default=1)
parser.add_argument('--fusion', dest='fusion',type=int, help ='if fusion 1 else 0',default=1)
parser.add_argument('--label', dest='label',type=str, help ='label in single task',default='ct')
parser.add_argument('--conv_archi', dest='conv_archi',type=int, help ='data_shape',default=4)
parser.add_argument('--tcn_archi', dest='tcn_archi',type=int, help ='tcn',default=0)

args = parser.parse_args()
d1 = read_rsdata.prepare_input_data(None, data = 'hx', conv = '3D', split = 'kfold')
d2 = read_rsdata.prepare_input_data(None, data = 'mx', conv = '2D', split = 'kfold')
hx,cg,ct = d1()
mx,_,_ = d2()
nSamples = 779
fusion,multitask = args.fusion,args.multitask
kfold = StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
ID_Inp = np.array(range(nSamples))
fold_no = 1
acc1_per_fold = []
acc2_per_fold =[]
loss1_per_fold =[]
loss2_per_fold =[]
ct,cg = np.array(ct),np.array(cg)
tf.random.set_seed(1234)

for IDs_Train, IDs_Test in kfold.split(ID_Inp, ct):
    hx_train,mx_train = hx[IDs_Train],mx[IDs_Train]
    ct_train, cg_train = ct[IDs_Train], cg[IDs_Train]
    ct_train = np_utils.to_categorical(ct_train,3) ##one hot encoding of crop type
    cg_train = np_utils.to_categorical(cg_train,3) ##one hot encoding of crop type
    hx_test,mx_test = hx[IDs_Test],mx[IDs_Test]
    ct_test, cg_test = ct[IDs_Test], cg[IDs_Test]
    ct_test = np_utils.to_categorical(ct_test,3) 
    cg_test = np_utils.to_categorical(cg_test,3)
    if args.conv_archi == 1:
        x_train,x_test = mx_train,mx_test
        neural_network = model.Convolution.Archi_1DCONV(x_train[0,:,:], 3, multitask)
    if args.conv_archi == 2:
        x_train,x_test = mx_train,mx_test
        neural_network = model.Convolution.Archi_2DCONV(x_train[0,:,:,:], 3, multitask)
    if args.conv_archi == 3:
        x_train,x_test = hx_train.reshape(-1,33,39,30,1),hx_test.reshape(-1,33,39,30,1)   
        neural_network = model.Convolution.Archi_3DCONV(x_train[0,:,:,:,:], 3,multitask)
    if args.conv_archi == 4:
        x_train,x_test = hx_train.reshape(-1,33,39,30,1),hx_test.reshape(-1,33,39,30,1)
        #x_train,x_test = hx_train,hx_test
        neural_network = tcn_.Conv2D_3D_TCN(x_train[0,:,:,:,:].shape,mx_train[0,:,:,:].shape,multitask,fusion)
    if args.conv_archi == 5:
        x_train,x_test = hx_train.reshape(-1,33,39,30,1),hx_test.reshape(-1,33,39,30,1)
        neural_network = tcn_.Conv1D_2D_TCN(x_train[0,:,:,:,:].shape,mx_train[0,:,:,:].shape,multitask,fusion)
    if args.tcn_archi == 1:
        x_train,x_test = hx_train,hx_test
        neural_network = tcn_.model((30,1287),(5,4096),multitask)
    if fusion:
        x_train_1,x_test_1 = x_train,x_test #changed according to architecture
        x_train_2,x_test_2 = mx_train,mx_test #changed according to architecture
    
    
        x_train = [x_train_1,x_train_2]
        x_test = [x_test_1,x_test_2]
    tf.random.set_seed(1234)

    if multitask==1:
        
        scores = multi_task(x_train, ct_train, cg_train, cg_test, ct_test,x_test)
        acc1_per_fold.append(scores[3])
        acc2_per_fold.append(scores[4])
        print(scores)
        loss1_per_fold.append(scores[1])
        loss2_per_fold.append(scores[2])
        fold_no += 1
        print(neural_network.summary())
        del neural_network,scores,x_train,ct_train,cg_train,x_test,cg_test,ct_test,hx_train,hx_test,mx_train,mx_test
        gc.collect()
    else:
        y = [cg_train,cg_test] if args.label == 'cg' else [ct_train,ct_test]
        scores = singletask(x_train,y[0],y[1],x_test)
        print(scores)
        acc1_per_fold.append(scores[1])
        loss1_per_fold.append(scores[0])
        fold_no += 1
        del neural_network,scores,x_train,ct_train,cg_train,x_test,cg_test,ct_test,hx_train,hx_test,mx_train,mx_test,y
        gc.collect()
if multitask==1:
    multitask_results(acc1_per_fold, loss1_per_fold, acc2_per_fold, loss2_per_fold)
else:
    singletask_results(acc1_per_fold, loss1_per_fold,args.label)



#np.savetxt('conv2d_3d_mx.csv',  (np.mean(acc1_per_fold), np.std(acc1_per_fold), np.mean(acc2_per_fold),np.std(acc2_per_fold)), delimiter=',')
#np.savetxt('conv2d_3d_hx_mx_ct.csv',  (np.mean(acc1_per_fold), np.std(acc1_per_fold)), delimiter=',')


