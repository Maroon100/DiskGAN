import os
import pickle
import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
import matplotlib
import configparser
import copy

path = '../data'
filename = 'failed.pkl'
datasetname = 'dataset.pkl'
complexity = 10
repeat = 3
limit = 50

def config_parse(section,key):
    cf = configparser.ConfigParser()
    cf.read('../configuration.ini')
    value = cf.get(section,key)
    return value

def filter(dataset,drop_index):
    drop_index = [str(i) for i in drop_index]
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

def polyfit(dataset,base,target,complexity):
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
    return f,score,y_pred,y_test

def get_cluster(disk_model):
    cluster = config_parse(disk_model,'cluster').split(';')
    cluster_num = len(cluster)
    for i in range(cluster_num):
        temp = [int(i) for i in cluster[i].split(',')]
        cluster[i] = temp
    return cluster

def fit(dataset,base,targets):
    errors = []
    for target in targets:
        sum = 0
        for i in range(repeat):
            _,score,_,_ = polyfit(dataset,base,target,complexity)
            sum += score
        errors.append(sum/repeat)
    return np.mean(errors)

def base_selection(dataset,disk_model):
    base_features = []
    clusters = get_cluster(disk_model)
    for cluster in clusters:
        index = 0
        cluster_size = len(cluster)
        errors = []
        for index in range(cluster_size):
            temp_cluster = copy.deepcopy(cluster)
            base = cluster[index]
            temp_cluster.pop(index)
            error = fit(dataset,base,temp_cluster)
            errors.append(error)
        minimum_error = min(errors)
        optimal_base = errors.index(minimum_error)
        base_features.append(cluster[optimal_base])
    print('The base feature(s) are: ')
    count = 1
    for base_feature in base_features:
        print("Cluster ",count,":",base_feature)
        count += 1

def cluster_plot(disk_model,dataset,base,targets,complexity,index):
    f = plt.figure(figsize=(9,9),dpi=80)
    row = len(targets)//2 + 1
    col = 2
    count = 0
    for target in targets:
        f,_,y_pred,y_test = polyfit(dataset,base,target,complexity)
        length = y_pred.shape[0]
        ax = plt.subplot(row,col,count+1)
        p1, = ax.plot(range(limit),y_pred[:limit],color='r',linestyle='--',marker='x',linewidth=1,label='pred')
        p2, = ax.plot(range(limit),y_test[:limit],color='g',linestyle='--',marker='+',linewidth=1,label='real')
        plt.ylim(0,1)
        plt.legend([p1,p2],['pred','real'],loc='best')
        plt.xlabel('Timeline(day)')
        plt.ylabel('Normalized Value of SMART Attributes')
        count +=1
    myfig = plt.gcf()
    plt.show()
    myfig.savefig(os.path.join('./',disk_model+'_cluster_'+str(index+1)+'.svg'),format='svg')
    print('The fitting result of cluster %d has been successfully saved!!!' % (index+1))

def plot(dataset,disk_model):
    base_features = config_parse(disk_model,'base_features')
    if len(base_features) == 1:
        base_features = [int(base_features)]
    else:
        base_features = [int(i) for i in base_features.split(',')]
    clusters = get_cluster(disk_model)
    count = 0
    for cluster in clusters:
        base_feature = base_features[count]
        base_index = cluster.index(base_feature)
        cluster.pop(base_index)
        print('base_feature: ',base_feature)
        print('target_features: ',cluster)
        cluster_plot(disk_model,dataset,base_feature,cluster,complexity,count)
        count += 1

def main():
    disk_model = 'Disk_1'
    file = open(os.path.join(path,disk_model,datasetname),'rb')
    dataset = pickle.load(file)
    drop_index = [int(i) for i in config_parse(disk_model,'drop_index').split(',')]
    dataset = filter(dataset,drop_index)
    dataset = normalize(dataset)['failed']['X']
    base_selection(dataset,disk_model)
    plot(dataset,disk_model)

if __name__ == '__main__':
	main()