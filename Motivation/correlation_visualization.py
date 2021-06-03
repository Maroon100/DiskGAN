import os
import pickle
import pandas as pd 
import matplotlib.pyplot as plt
import configparser

path = '../data/Disk_4'
filename = 'failed.pkl'
pic_path = './'
index = 0
clusters = [[0,1,9],[2,3,10,11],[4,5,7,8]]
color_list = ['g', 'b', 'r', 'c', 'm', 'k', 'w', 'y']
marker_list = ['.','o','v','^','x','+']
limit = 50
upper_bound = [1,1,1]

def config_parse(section,key):
	cf = configparser.ConfigParser()
	cf.read('../configuration.ini')
	value = cf.get(section,key).split(',')
	return value

def plot(cluster,feature_dict,sample,index):
	length = sample.shape[0]
	count = 0
	p_list = []
	labels = []
	font = {'family' : 'Times New Roman',
    'weight' : 'bold',
    }
	for attribute in cluster:
		attr_list = list(sample[feature_dict[str(attribute)]])
		p, = plt.plot(range(limit),attr_list[:limit],linestyle='--',marker=marker_list[count],c=color_list[count],linewidth = 2,label=feature_dict[str(attribute)])
		p_list.append(p)
		labels.append(feature_dict[str(attribute)])
		count += 1
	plt.ylim(0,upper_bound[index])
	plt.legend(p_list,labels,loc='best')
	plt.xlabel('Timeline(day)',font)
	plt.ylabel('Normalized Value of SMART Attributes',font)
	myfig = plt.gcf()
	plt.show()
	myfig.savefig(os.path.join(pic_path,'cluster_'+str(index)+'.svg'),format='svg')
	print('Correlations of cluster %d have been successfully saved!!!'%(index+1))

def main():
	file = open(os.path.join(path,filename),'rb')
	data = pickle.load(file)
	sample = data[index]
	sample['date'] = pd.to_datetime(sample['date'])
	sample = sample.sort_values('date')
	columns = sample.columns
	sample = sample.drop(columns=columns[:6])
	sample = sample.drop(columns=config_parse('Disk_4','drop_list'))
	sample = (sample - sample.min())/(sample.max()-sample.min())
	feature_dict = {}
	count = 0
	for attribute in sample.columns:
		feature_dict[str(count)] = attribute
		count += 1
	count = 0
	for cluster in clusters:
		plot(cluster,feature_dict,sample,count)
		count += 1

if __name__ == '__main__':
	main()