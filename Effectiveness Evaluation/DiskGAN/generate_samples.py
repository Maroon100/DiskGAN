import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
import copy
from keras.models import load_model
import tensorflow as tf
import random
import time
# import filter
import configparser

feature_cluster = None
base_feature = None
target = None
complexity = 10
SMART_ATTRIBUTE = 999
FEATURES = -1
SAMPLE_NUM = -1
THRESHOLD = 0.3

def config_parse(section,key):
    cf = configparser.ConfigParser()
    cf.read('../configuration.ini')
    value = cf.get(section,key).split(',')
    return value

def get_cluster(disk_model):
    cf = configparser.ConfigParser()
    cf.read('../configuration.ini')
    cluster = cf.get(disk_model,'cluster').split(';')
    cluster_num = len(cluster)
    for i in range(cluster_num):
        temp = [int(i) for i in cluster[i].split(',')]
        cluster[i] = temp
    return cluster

def loaddata(records,base,target):
    data = []
    label = []
    sample_len = len(records[0])
    feature_len = len(records[0][0])
    for record in records:
        record = pd.DataFrame(record,columns=[str(i) for i in range(feature_len)],index=[i for i in range(sample_len)])
        data.append(record[str(base)].to_numpy())
        label.append(record[str(target)].to_numpy())
    data = np.asarray(data)
    data = data.reshape((data.shape[0],data.shape[1],1))
    label = np.asarray(label)
    label = label.reshape((label.shape[0],label.shape[1],1))
    return data,label

def polyfit(dataset,base,target):
    data,label = loaddata(dataset,base,target)
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=None)
    x_train = x_train.reshape((x_train.shape[0] * x_train.shape[1]))
    y_train = y_train.reshape((y_train.shape[0] * y_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0] * x_test.shape[1]))
    y_test = y_test.reshape((y_test.shape[0] * y_test.shape[1]))
    f = np.polyfit(x_train, y_train, complexity)
    f = np.poly1d(f)
    y_pred = f(x_test)
    score = mean_squared_error(y_test, y_pred)
    return f,score

def pre_fit(disk_model,path):
    file = open(os.path.join(path,disk_model,'samples.pkl'),'rb')
    train = pickle.load(file)['X']
    func_dict = {}
    for feature in base_feature:
        func_dict[str(feature)] = []
    for key in target.keys():
        for value in target[key]:
            f,score = polyfit(train,int(key),value)
            func_dict[key].append(f)
    return func_dict

def load_base_feature(path='./',basename='gene_samples_'):
    dataset = []
    for i in base_feature:
        filename = basename + str(i) + '.pkl'
        file = open(os.path.join(path,filename),'rb')
        data = pickle.load(file).tolist()[:12]
        dataset.append(data)
    return dataset

def random_group(base_features):
    base_dataset = []
    length = len(base_features)
    count = 0
    for i in range(length):
        temp_dataset = []
        if count == 0:
            temp_dataset.extend(base_features[i])
        elif count == 1:
            for m in base_dataset:
                for j in base_features[i]:
                    temp_dataset.append([m,j])
        else:
            for m in base_dataset:
                for j in base_features[i]:
                    temp = copy.deepcopy(m)
                    temp.append(j)
                    temp_dataset.append(temp)
        count += 1
        base_dataset = copy.deepcopy(temp_dataset)
    for i in range(len(base_dataset)):
        for j in range(len(base_dataset[0])):
            base_dataset[i][j] = np.asarray(base_dataset[i][j])/SMART_ATTRIBUTE
    return base_dataset

def get_base(tar):
    base, index = -1,-1
    for key in target.keys():
        if tar in target[key]:
            base = key
            index = target[key].index(tar)
            break
    return base,index


def get_samples(func_dict,base_sample):
    samples = {}
    features = range(0,FEATURES)
    count = 0
    data = []
    for i in base_feature:
        samples[str(i)] = base_sample[count]
        count += 1
    for i in features:
        if i in base_feature:
            continue
        else:
            base,index = get_base(i)
            pred = func_dict[base][index](samples[base])
            samples[str(i)] = pred
    for i in features:
        data.append(samples[str(i)])
    return np.asarray(data)

def gene_samples(func_dict,base_features):
    base_group = random_group(base_features)
    if len(base_feature)==1:
        base_group = [[i] for i in base_group]
    dataset = []
    for base_sample in base_group:
        samples = get_samples(func_dict,base_sample)
        dataset.append(samples.T)
    return np.asarray(dataset)

def metric_FAR(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    FAR=FP/(TN+FP)
    return FAR

def metric_FDR(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    FDR=TP/(TP+FN)
    return FDR

def metric_F1score(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1score=2*precision*recall/(precision+recall)
    return F1score

def get_final_samples(disk_model,path,savepath,modelpath,basepath):
    global SAMPLE_NUM
    func_dict = pre_fit(disk_model,path)
    base_features = load_base_feature(path=basepath)
    samples = gene_samples(func_dict,base_features)
    model = load_model(os.path.join(modelpath, disk_model+'_filter.h5'),
                       custom_objects={"metric_F1score": metric_F1score, "metric_FDR": metric_FDR,
                                       "metric_FAR": metric_FAR})
    prediction = model.predict(samples)
    selected = []
    for i in range(samples.shape[0]):
        if prediction[i][0]>THRESHOLD:
            selected.append(samples[i])

    random.shuffle(selected)
    qualified_num = len(selected)
    print('qualified_num: ',qualified_num)
    SAMPLE_NUM = SAMPLE_NUM if SAMPLE_NUM<qualified_num else qualified_num
    selected = selected[:SAMPLE_NUM]
    labels = [[1]] * SAMPLE_NUM
    generated = {'X':selected,'Y':labels}
    print('selected',len(selected))
    print('labels: ',len(labels))
    file = open(os.path.join(savepath,disk_model+'_diskgan.pkl'),'wb')
    pickle.dump(generated,file)
    print('Generated samples have been successfully saved !!!')

def gen(disk_model,datapath,savepath,modelpath,basepath,gen_num,threshold):
    global feature_cluster, base_feature, target, FEATURES, SAMPLE_NUM, THRESHOLD
    base_feature = [int(i) for i in config_parse(disk_model,'base_features')]
    temp_cluster = get_cluster(disk_model)
    count = 1
    sum = 0
    feature_cluster = {}
    target = {}
    for cluster in temp_cluster:
        feature_cluster[str(count)] = cluster
        sum += len(cluster)
        cluster.remove(base_feature[count-1])
        target[str(base_feature[count-1])] = cluster
        count += 1
    FEATURES = sum
    SAMPLE_NUM = gen_num
    THRESHOLD = threshold
    get_final_samples(disk_model,datapath,savepath,modelpath,basepath)

if __name__ == '__main__':
    gen()