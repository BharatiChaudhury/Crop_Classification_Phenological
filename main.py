import argparse
import read_rsdata
import model
import tensorflow as tf
import tcn_
import utils


parser = argparse.ArgumentParser(description='Running ConvTCN architectures')
parser.add_argument('--read_data', dest='read_data',type=str, help ='read data',default=1)
parser.add_argument('--read_data_conv2d_3d', dest='read_data_conv2d_3d',type=str, help ='read_data conv_2d_3d',default=1)

parser.add_argument('--conv_archi', dest='conv_archi',type=int, help ='data_shape',default=3)
parser.add_argument('--multitask', dest='multitask',type=int, help ='if multitask 1 else 0',default=1)
parser.add_argument('--fusion', dest='fusion',type=int, help ='if fusion 1 else 0',default=1)

parser.add_argument('--tcn_archi', dest='tcn_archi',type=int, help ='tcn',default=0)
parser.add_argument('--label', dest='label',type=str, help ='label in single task',default='ct')


args = parser.parse_args()
if args.read_data_conv2d_3d == 1:
    d1 = read_rsdata.data_shape(num=1)
    d2 = read_rsdata.data_shape(num=4)
    
    mx_train,cg_train,ct_train,mx_val,cg_val,ct_val,mx_test,cg_test,ct_test = d1() #returns data -> input trainig data, y1 -> #crop type, y2 -> crop growth stage
    hx_train,cg_train,ct_train,hx_val,cg_val,ct_val,hx_test,cg_test,ct_test = d2() #returns data -> input trainig data, y1 -> #crop type, y2 -> crop growth stage

else:
    nums = args.read_data
    d = read_rsdata.data_shape(num = nums)

    if nums in [6,7,8]:
        hx_train,mx_train,cg_train,ct_train,hx_val,mx_val,cg_val,ct_val,hx_test,mx_test,cg_test,ct_test =d()  
    elif nums in [1,2,0]:
        mx_train,cg_train,ct_train,hx_val,cg_val,ct_val,mx_test,cg_test,ct_test = d() #returns data -> input trainig data, y1 -> #crop type, y2 -> crop growth stage
        
    else:
        hx_train,cg_train,ct_train,hx_val,cg_val,ct_val,hx_test,cg_test,ct_test = d() #returns data -> input trainig data, y1 -> #crop type, y2 -> crop growth stage

tf.random.set_seed(1234)

multitask = args.multitask
fusion = args.fusion
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

if args.conv_archi == 1:

    x_train,x_test = mx_train,mx_test
    neural_network = model.Convolution.Archi_1DCONV(x_train[0,:,:], 3, multitask)
if args.conv_archi == 2:
    
    x_train,x_test = hx_train.reshape(-1,33,39,30,1),hx_test.reshape(-1,33,39,30,1)   
    neural_network = model.Convolution.Archi_3DCONV(x_train[0,:,:,:,:], 3,multitask)

if args.conv_archi == 3:
    
    x_train,x_test = hx_train.reshape(-1,33,39,30,1),hx_test.reshape(-1,33,39,30,1)   
    neural_network = tcn_.Conv2D_3D_TCN(x_train[0,:,:,:,:].shape,mx_train[0,:,:,:].shape,multitask,fusion)
    

if args.tcn_archi == 1:
    if fusion:
        neural_network = tcn_.model((5,4096),(30,1287),multitask)
    else:
        neural_network = tcn_.mx_tcn_model((5,4096), multitask)

def compute_cost(Y_pred,Y_label):
    cce = tf.keras.losses.categorical_crossentropy()
    loss = (Y_label,Y_pred).numpy()
    return loss
def joint_loss(cg_pred,ct_train,ct_loss,ct_pred,alpha):
    cg_loss = compute_cost(cg_pred, cg_train)
    ct_loss = compute_cost(ct_pred, ct_train)
    joint_loss = alpha*ct_loss+(1-alpha)*cg_loss
    return joint_loss




def multi_task(x_train,ct_train,cg_train,cg_test,ct_test,x_test):
    loss_list = {'cg':'categorical_crossentropy','ct':'categorical_crossentropy'}
    metric_list = {'ct':'accuracy','cg':'accuracy'}
    loss_weights = {'cg':0.3,'ct':0.7}

    neural_network.compile(loss = loss_list,optimizer = opt, metrics = metric_list, loss_weights=loss_weights)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=10)
    if fusion:
        history_cnn = neural_network.fit(x = [x_train[0],x_train[1]], y = [ct_train,cg_train], batch_size=25,epochs=50,validation_data=([x_test[0],x_test[1]],[ct_test,cg_test]),verbose=0)
        print(neural_network.evaluate([x_train[0],x_train[1]], [ct_train,cg_train], batch_size=20,verbose=0))
        print(neural_network.evaluate([x_test[0],x_test[1]], [ct_test,cg_test], batch_size=20,verbose=0))
        results = neural_network.predict([x_test[0],x_test[1]])
        #print(results)
        utils.plot_multi_task_loss(history_cnn)
        utils.precision_recall_f1score(ct_test,cg_test, results)
        return results


    else:
        history_cnn = neural_network.fit(x = x_train, y = [ct_train,cg_train], batch_size=25,epochs=50,verbose=0)
        print(neural_network.evaluate(x_train, [ct_train,cg_train], batch_size=20,verbose=0))
        print(neural_network.evaluate(x_test, [ct_test,cg_test], batch_size=20,verbose=0))
    

def singletask(x_train,y_train,y_test,x_test):
    neural_network.compile(loss = 'categorical_crossentropy',optimizer = opt, metrics = ['accuracy'])

    if fusion:
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=6)
        
        history_cnn = neural_network.fit(x = [x_train[0],x_train[1]], y = y_train, validation_data=([x_test[0],x_test[1]],y_test),batch_size=25,epochs=50,verbose=0)
        print(neural_network.evaluate([x_test[0],x_test[1]], y_test, batch_size=20,verbose=0))
        utils.plot_single_task_loss(history_cnn,args.label)

    else:
        history_cnn = neural_network.fit(x = x_train, y = y_train, batch_size=25,epochs=50,verbose=0)
        print(neural_network.evaluate(x_train, y_train, batch_size=20,verbose=0))
        print(neural_network.evaluate(x_test, y_test, batch_size=20,verbose=0))

results = []
    
if fusion == 1:
    x_train_1,x_test_1 = x_train,x_test #changed according to architecture
    x_train_2,x_test_2 = mx_train,mx_test #changed according to architecture
    x_train = [x_train_1,x_train_2]
    x_test = [x_test_1,x_test_2]
    if multitask==1:
        results = multi_task(x_train, ct_train, cg_train, cg_test, ct_test,x_test)
    else:
        y = [cg_train,cg_test] if args.label == 'cg' else [ct_train,ct_test]
        singletask(x_train,y[0],y[1],x_test)
        
else:
    if multitask==1:
        results = multi_task(x_train, ct_train, cg_train, cg_test, ct_test, x_test, results)
    else:
       y = [cg_train,cg_test] if args.label == 'cg' else [ct_train,ct_test]
       singletask(x_train,y[0],y[1],x_test)
        

