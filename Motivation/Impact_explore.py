import os
import pickle
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import SimpleRNN,Dense,GRU,Bidirectional
from keras.layers import BatchNormalization
from keras import regularizers
from keras import optimizers
import random
import pandas as pd
import configparser

path = '../data'
filename = 'dataset.pkl'
data_ratio = [6,8,10]
labeled_ratio = 0.1
screwness = 1
HIDDEN_DIM = 64
DROPOUT = 0.1
BATCHSIZE = 128
EPOCHS = 2000
first = 'failed'

drop_index = None

def config_parse(section,key):
    cf = configparser.ConfigParser()
    cf.read('../configuration.ini')
    value = cf.get(section,key)
    return value

def filter(dataset,disk_model):
    drop_index = config_parse(disk_model,'drop_index').split(',')
    sample_len = len(dataset['failed']['X'][0])
    feature_len = len(dataset['failed']['X'][0][0])
    for i in range(len(dataset['failed']['X'])):
        temp = dataset['failed']['X'][i]
        temp = pd.DataFrame(temp,columns=[str(i) for i in range(feature_len)],index=[str(i) for i in range(sample_len)])
        temp = temp.drop(columns=drop_index)
        temp = [temp.iloc[m].to_list() for m in range(temp.shape[0])]
        dataset['failed']['X'][i] = temp
    
    for i in range(len(dataset['healthy']['X'])):
        temp = dataset['healthy']['X'][i]
        temp = pd.DataFrame(temp,columns=[str(i) for i in range(feature_len)],index=[str(i) for i in range(sample_len)])
        temp = temp.drop(columns=drop_index)
        temp = [temp.iloc[m].to_list() for m in range(temp.shape[0])]
        dataset['healthy']['X'][i] = temp
    return dataset

def get_list(dataset):
    length = len(dataset[0][0])
    max_value_list, min_value_list = [0]*length, [float('inf')]*length
    for record in dataset:
        for item in record:
            for i in range(len(item)):
                if item[i] > max_value_list[i]:
                    max_value_list[i] = item[i]
                elif item[i] < min_value_list[i]:
                    min_value_list[i] = item[i]
    return max_value_list,min_value_list

def normalize(dataset):
    failed = dataset['failed']['X']
    failed_len = len(failed)
    healthy = dataset['healthy']['X']
    failed.extend(healthy)
    data = failed
    max_value_list, min_value_list = get_list(data)
    for record in data:
        for item in record:
            for i in range(len(item)):
                if (max_value_list[i] - min_value_list[i])!=0:
                    item[i] = (item[i] - min_value_list[i])/(max_value_list[i] - min_value_list[i])
                else:
                    item[i] /= max_value_list[i]
    dataset['failed']['X'] = data[:failed_len]
    dataset['healthy']['X'] = data[failed_len:]
    return dataset

def shuffle(x,y):
    randnum = random.randint(0,100)
    random.seed(randnum)
    random.shuffle(x)
    random.seed(randnum)
    random.shuffle(y)
    return x,y

def split(dataset,downsampling,screw):
    failed = dataset['failed']
    healthy = dataset['healthy']
    train,val,test = {'X':[],'Y':[]},{'X':[],'Y':[]},{'X':[],'Y':[]}
    failed_len = len(failed['X'])
    healthy_len = len(healthy['X'])

    failed['X'],failed['Y'] = shuffle(list(failed['X']),failed['Y'])
    healthy['X'],healthy['Y'] = shuffle(list(healthy['X']),healthy['Y'])
    
    train_failed_bound = int(failed_len * data_ratio[0] / 10)
    train_healthy_bound = int(healthy_len * data_ratio[0] / 10)    

    train_failed_len = int(failed_len * data_ratio[0] / 10 * labeled_ratio)
    train_healthy_len = int(healthy_len * data_ratio[0] / 10 * labeled_ratio)

    val_failed_len = int(failed_len * data_ratio[1] / 10 * labeled_ratio)
    val_healthy_len = int(healthy_len * data_ratio[1] / 10 * labeled_ratio)

    test_failed_len = int(failed_len * 0.2)
    test_healthy_len = int(healthy_len * 0.2)

    train['X'] = failed['X'][:train_failed_len] + healthy['X'][:train_healthy_len]
    train['Y'] = failed['Y'][:train_failed_len] + healthy['Y'][:train_healthy_len]

    val['X'] = failed['X'][train_failed_bound+train_failed_len:train_failed_bound+val_failed_len] + healthy['X'][train_healthy_bound+train_healthy_len:train_healthy_bound+val_healthy_len]
    val['Y'] = failed['Y'][train_failed_bound+train_failed_len:train_failed_bound+val_failed_len] + healthy['Y'][train_healthy_bound+train_healthy_len:train_healthy_bound+val_healthy_len]

    test['X'] = failed['X'][-test_failed_len:] + healthy['X'][-test_healthy_len:]
    test['Y'] = failed['Y'][-test_failed_len:] + healthy['Y'][-test_healthy_len:]

    val_failed_length = val_failed_len - train_failed_len

    if downsampling:
        train['X'] = train['X'][:2*train_failed_len]
        train['Y'] = train['Y'][:2*train_failed_len]
        val['X'] = val['X'][:2*val_failed_length]
        val['Y'] = val['Y'][:2*val_failed_length]
    if screw>0 and first=='healthy':
        train['X'] = train['X'][:(1+screwness)*train_failed_len]
        train['Y'] = train['Y'][:(1+screwness)*train_failed_len]
        val['X'] = val['X'][:(1+screwness)*val_failed_length]
        val['Y'] = val['Y'][:(1+screwness)*val_failed_length]
    if screw>0 and first == 'failed':
        train['X'] += failed['X'][train_failed_len:(1+screwness)*train_failed_len]
        train['Y'] += failed['Y'][train_failed_len:(1+screwness)*train_failed_len]
        val['X'] += failed['X'][train_failed_bound+train_failed_len:train_failed_bound+(1+screwness)*val_failed_length]
        val['Y'] += failed['Y'][train_failed_bound+train_failed_len:train_failed_bound+(1+screwness)*val_failed_length]

    train['X'],train['Y'] = shuffle(train['X'],train['Y'])
    val['X'],val['Y'] = shuffle(val['X'],val['Y'])
    test['X'],test['Y'] = shuffle(test['X'],test['Y'])

    train['X'], train['Y'] = np.asarray(train['X']), np.asarray(train['Y'])
    val['X'],val['Y'] = np.asarray(val['X']),np.asarray(val['Y'])
    test['X'],test['Y'] = np.asarray(test['X']),np.asarray(test['Y'])
    remain_failure = np.asarray(failed['X'][train_failed_len:])
    return train,val,test,remain_failure

def metric_F1score(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1score=2*precision*recall/(precision+recall)
    return F1score

def metric_FDR(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    FDR=TP/(TP+FN)
    return FDR

def metric_FAR(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    FAR=FP/(TN+FP)
    return FAR

def get_model():
    model = Sequential()
    model.add(Bidirectional(GRU(256,activation='relu',return_sequences=True,kernel_regularizer=regularizers.l2(l=0.001))))
    model.add(BatchNormalization())
    model.add(Bidirectional(GRU(128,activation='relu',return_sequences=True,kernel_regularizer=regularizers.l2(l=0.001))))
    model.add(BatchNormalization())
    model.add(GRU(10,activation='relu',kernel_regularizer=regularizers.l2(l=0.001)))
    model.add(BatchNormalization())
    model.add(Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l2(l=0.001)))
    return model

def explore(disk_model):
    file = open(os.path.join(path,disk_model,filename),'rb')
    dataset = pickle.load(file)
    dataset = filter(dataset,disk_model)
    dataset = normalize(dataset)
#    train,val,test = split(dataset,True,False)
    train,val,test,remain = split(dataset,False,True)
    model = get_model()
    opt = optimizers.Adam(lr=0.00007)
    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy',metric_F1score,metric_FDR,metric_FAR])

    model.fit(train['X'],train['Y'],batch_size=BATCHSIZE,epochs=EPOCHS,validation_data=(val['X'],val['Y']),verbose=True)
    score,acc,f1,FDR,FAR = model.evaluate(test['X'],test['Y'],steps=1,verbose=0)
    print('Test score: ',score)
    print('Test accuracy: ',acc)
    print('True Positive Rate: ',round(FDR,3))
    print('False Alarm Rate: ',round(FAR,3))
    print('F1 Score: ',round(f1,3))

def main():
    disk_model = 'Disk_1'
    explore(disk_model)

if __name__=='__main__':
    main()

