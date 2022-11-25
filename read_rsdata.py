# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 15:19:03 2022

@author: bharathi
"""
import os
import xarray as xr
import numpy as np
from sklearn.model_selection import train_test_split        
from keras.utils import np_utils
import utils
class prepare_input_data:
    
    def __init__(self,path,**kwargs):
        self.path = path
        self.kwargs = kwargs
        
    def read_data(self):
        path = self.path
        kwargs = self.kwargs
        stl_cg = kwargs.setdefault('stl_cg', None)
        if path is None:
            p = "E:\\Crop_Classification_Phenological\\UAS Precision NCAT files"
        else:
            p = path
        mx_crop_growth_stage = ["2020-06-30_MSI_Sorted", "2020-07-23_MSI_Sorted", "2020-09-03_MSI_Sorted"]
        hx_crop_growth_stage = ["2020-06-30_HSI_Sorted", "2020-07-23_HSI_Sorted", "2020-09-04_HSI_Sorted"]
        ##Mx or Hx of Hx-Mx Input for single task crop growth stage__##
        if(stl_cg==1):
            if(kwargs['data']=='hx'):
                training = self.create_single_task_crop_growth_stage(hx_crop_growth_stage, p)
            elif(kwargs['data']=='mx'):
                training = self.create_single_task_crop_growth_stage(mx_crop_growth_stage,p)
            else:
                training_hx,training_mx = self.create_single_task_cg_training_data_hx_mx(hx_crop_growth_stage,mx_crop_growth_stage, p)
                training1,training2 = training_hx[0:215],training_hx[216:781] ##Discarding both mx and hx input where the input shape of hx is invalid 
                training_hx = training1+training2
                training1,training2 = training_mx[0:215],training_mx[216:781]
                training_mx = training1+training2
                return training_hx,training_mx
            training1 = training[0:215] ##Discarding both mx and hx input where the input shape of hx is invalid 
            training2 = training[216:781]
            training = training1+training2
        ##__Mx or Hx or Hx-Mx Input for MTL task ##
        else:
            if(kwargs['data']=='hx'):
                training = self.create_training_data(hx_crop_growth_stage, p)
            elif(kwargs['data']=='mx'):
                training = self.create_training_data(mx_crop_growth_stage,p)
            else:
                training_hx,training_mx = self.create_training_data_hx_mx(hx_crop_growth_stage,mx_crop_growth_stage, p)
                training1,training2 = training_hx[0:215],training_hx[216:781] ##Discarding both mx and hx input where the input shape of hx is invalid 
                training_hx = training1+training2
                training1,training2 = training_mx[0:215],training_mx[216:781]
                training_mx = training1+training2
                return training_hx,training_mx
            training1 = training[0:215] ##Discarding both mx and hx input where the input shape of hx is invalid 
            training2 = training[216:781]
            training = training1+training2
        return training
    
    ##__Function for either hx or mx data__
    def create_training_data(self,crop_growth_stage,p):
        training = []
        crop_type = ["corn", "cotton", "soybean"]
        for cg in crop_growth_stage:
            class_1 = crop_growth_stage.index(cg) # y_1-> crop growth stage
            for ct in crop_type:
                path = os.path.join(p,cg,ct)
                class_2 = crop_type.index(ct) # y_2 -> crop type
                for img in os.listdir(path+"\\No_Geo"):
                    raster_array = xr.open_rasterio(os.path.join(path+"\\No_Geo",img))
                    data = raster_array.values
                    training.append([data, class_1,class_2])
        return training
    ##__Function for single task crop growth stage__#
    def create_single_task_crop_growth_stage(self,crop_growth_stage,p):
        i=0
        training = []
        crop_type = ["corn", "cotton", "soybean"]
        for cg in crop_growth_stage:
            for ct in crop_type:
                path = os.path.join(p,cg,ct)
                label = i
                print(path,label)
                for img in os.listdir(path+"\\No_Geo"):
                    raster_array = xr.open_rasterio(os.path.join(path+"\\No_Geo",img))
                    data = raster_array.values
                    training.append([data, label])
            i+=1
        return training
    ##__Function for fusion of hx and mx data for crop growth stage
    def create_single_task_cg_training_data_hx_mx(self,crop_growth_stage_hx,crop_growth_stage_mx,p):
        training_hx = self.create_single_task_crop_growth_stage(crop_growth_stage_hx,p)
        training_mx = self.create_single_task_crop_growth_stage(crop_growth_stage_mx,p)
        return training_hx,training_mx
    
    ##__Function for fusion of hx and mx data
    def create_training_data_hx_mx(self,crop_growth_stage_hx,crop_growth_stage_mx,p):
        training_hx = self.create_training_data(crop_growth_stage_hx,p)
        training_mx = self.create_training_data(crop_growth_stage_mx,p)
        return training_hx,training_mx
    
    ##Function for reshape of data for input into Convolution Architecture
    def reshape_data(self, X):
            kwargs = self.kwargs
            X = np.array(X)
            _, d,w,h = X.shape
            if(d!=5):
                w,h,d = d,w,h
            if(kwargs['conv']=='1D'):
                X = X.reshape(-1,d,w*h)
            elif(kwargs['conv']=='2D' and d==5):
                X = np.transpose(X,(0,2,3,1))
            elif(kwargs['conv']=='2D' and d!=5):
                return X
            else:
                X = X.reshape(-1,d,w,h,1)
            return X
    
    ##___Function to read hyperspectral data with pca and without pca__##   
    def read_hx(self,training):
        X,y_1,y_2 = [],[],[]
        kwargs = self.kwargs
        pca = kwargs.setdefault("PCA",1)
        for features, label1, label2 in training:
            if features.shape[0]==270:
                if pca==1: 
                    inp_ = np.transpose(features[0:220,0:33,0:39],(1,2,0))
                    X.append(utils.PCA_(inp_))
                
                elif pca==0: 
                    X.append(np.transpose(features[0:220,0:33,0:39],(1,2,0))) ##input shape of all hyperspectral shape is not same; this ensures the extraction of common hx pixel values
            
                y_1.append(label1)
                y_2.append(label2)
        X = self.reshape_data(X)
        return X,y_1,y_2
    
    ##__Function to read Multispectral data__#
    def read_mx(self,training):
        X,y_1,y_2 = [],[],[]
        for features, label1, label2 in training: 
            X.append(features) ## extraction of mx pixel values
            y_1.append(label1)
            y_2.append(label2)
        X = self.reshape_data(X)
        return X,y_1,y_2
    
    ##__Function to create window patches of same size for hyperspectral and multispectral__#
    def create_image_patches(self,training):
        X = utils.createImageCubes(training,5)
        return X
    
    ##__Prepare training data__##
    def __call__(self):
        kwargs = self.kwargs
        hx,mx = [],[]
        split = kwargs.setdefault('split',None)
        if(kwargs['data']=='hx' or kwargs['data']=='mx'):
            training = self.read_data()
            if(kwargs['data']=='hx'):
                hx,y_1,y_2 = self.read_hx(training)
            else:
                mx,y_1,y_2 = self.read_mx(training)
            X = hx if len(hx)>0 else mx   
            
            if(split=='kfold'):
                return X,y_1,y_2
            else:
               ##train test split
               y_1 = np_utils.to_categorical(y_1,3) ##one hot encoding of crop growth stage
               y_2 = np_utils.to_categorical(y_2,3) ##one hot encoding of crop type
               X_train, X_test, y_train, y_test = train_test_split(X, y_1, test_size = 0.2, random_state = 42)
               X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=0)
        
               X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y_2, test_size=0.20, random_state=42)
               X_train_2, X_val_2, y_train_2, y_val_2 = train_test_split(X_train_2, y_train_2, test_size=0.20, random_state=0)
               return X_train,y_train,y_train_2,X_val,y_val,y_val_2,X_test,y_test,y_test_2

        else:
            training_hx,training_mx = self.read_data()
            hx,y_1,y_2 = self.read_hx(training_hx)
            mx,y_1,y_2 = self.read_mx(training_mx)
            
            y_1 = np_utils.to_categorical(y_1,3) ##one hot encoding of crop growth stage
            y_2 = np_utils.to_categorical(y_2,3) ##one hot encoding of crop type
            
            
            X_train, X_test, y_train, y_test = train_test_split(hx, y_1, test_size = 0.2, random_state = 42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=0)
        
            X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(hx, y_2, test_size=0.20, random_state=42)
            X_train_2, X_val_2, y_train_2, y_val_2 = train_test_split(X_train_2, y_train_2, test_size=0.20, random_state=0)
            
            X_train_3, X_test_3, y_train, y_test = train_test_split(mx, y_1, test_size = 0.2, random_state = 42)
            X_train_3, X_val_3, y_train, y_val = train_test_split(X_train_3, y_train, test_size=0.20, random_state=0)
        
            X_train_4, X_test_4, y_train_2, y_test_2 = train_test_split(mx, y_2, test_size=0.20, random_state=42)
            X_train_4, X_val_4, y_train_2, y_val_2 = train_test_split(X_train_4, y_train_2, test_size=0.20, random_state=0)
    
            return X_train,X_train_3,y_train,y_train_2,X_val,X_val_3,y_val,y_val_2,X_test,X_test_3,y_test,y_test_2

def data_shape(**kwargs):
    switcher ={
        0 : prepare_input_data(None, data='mx',conv = '1D'),
        1 : prepare_input_data(None, data='mx',conv = '2D'),
        2 : prepare_input_data(None, data='mx',conv = '3D'),
        
        3 : prepare_input_data(None, data='hx',conv = '1D'),
        4 : prepare_input_data(None, data='hx',conv = '2D'),
        5 : prepare_input_data(None, data='hx',conv = '3D'),

        6 : prepare_input_data(None, data='hx_mx',conv = '1D'),
        7 : prepare_input_data(None, data='hx_mx',conv = '2D'),
        8 : prepare_input_data(None, data='hx_mx',conv = '3D')
    }
    return(switcher.get(kwargs['num']))
        
#d = prepare_input_data(None, data='hx_mx',conv = '2D')
#X_train,y_train,y_train_2,X_val,y_val,y_val_2,X_test,y_test,y_test_2 = d() #returns data -> input trainig data, y1 -> 
#hx_train,mx_train,cg_train,ct_train,hx_val,mx_val,cg_val,ct_val,hx_test,mx_test,cg_test,ct_test =d() #crop type, y2 -> crop growth stage 