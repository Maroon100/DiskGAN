import os
import pickle
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import configparser

path = '../data'
filename = 'failed.pkl'
datasetname = 'dataset.pkl'
threshold = 0.5
pic_path = './'

attribute_map = {
	'smart_1_normalized':1,
	'smart_1_raw':2,
	'smart_4_raw':3,
	'smart_7_raw':4,
	'smart_9_raw':5,
	'smart_12_raw':6,
	'smart_190_normalized':7,
	'smart_190_raw':8,
	'smart_192_raw':9,
	'smart_193_raw':10,
	'smart_194_normalized':11,
	'smart_194_raw':12,
	'smart_195_normalized':13,
	'smart_195_raw':14,
	'smart_241_raw':15,
	'smart_242_raw':16
}

ssd_map = {                          # The 'n' represents normalized and 'r' means raw. Thus the 'n_9' is actually 'smart_9_normalized'
	'n_9': 1,
	'r_9': 2,
	'n_173': 3,
	'r_175': 4,
	'r_177': 5,
	'n_190': 6,
	'r_190': 7,
	'r_194': 8,
	'r_195': 9,
	'r_241': 10
}

def config_parse(section,key):
	cf = configparser.ConfigParser()
	cf.read('../configuration.ini')
	value = cf.get(section,key).split(',')
	return value

def find_same(record):
	drop_list = []
	columns = record.columns
	for column in columns:
		series = record[column]
		number = series.iloc[0]
		same = True
		for i in range(series.size):
			if number != series.iloc[i]:
				same = False
				break
		if same:
			drop_list.append(column)
	return drop_list

def judge(result,drop_list):
	drop_set = set(drop_list)
	flag = True
	for r in result:
		if r not in drop_set:
			flag = False
			break
	return flag

def map(sum,is_ssd):
	attributes = []
	mapper = ssd_map if is_ssd else attribute_map
	for attribute in sum.columns:
		attributes.append(str(mapper[attribute]))
	sum.columns = pd.Index(attributes)
	sum.index = pd.Index(attributes)

def corr_plot(disk_model,matrix):
	column_len = matrix.columns.size
	f,ax = plt.subplots(figsize=(column_len/2,column_len/2))
	sns.set(context='paper',font_scale=1.0)
	ax = sns.heatmap(matrix,cmap='YlGnBu',linewidths=0.05,ax=ax,square=True,annot=True)
	ax.tick_params(labelsize=10)
	ax.set_ylim(column_len,0)
	ax.set_xlim(column_len,0)
	ax.xaxis.tick_top()
	plt.yticks(size=10)
	plt.xticks(size=10,rotation=90)
	plt.subplots_adjust(top=0.5,right=0.8)
	myfig = plt.gcf()
	plt.show()
	myfig.savefig(os.path.join(pic_path,disk_model+'_corr.svg'),format='svg')
	print('The correlation matrix has been successfully saved!!!')

def find_cluster(disk_model,matrix):
	columns = matrix.columns.to_list()
	clusters = []
	flags = [0]*len(columns)
	count = 0
	for column in columns:
		if flags[count]==1:
			count += 1
			continue
		correlations = matrix[column]
		tmp_count = 0
		temp_cluster = []
		temp_cluster.append(column)
		flags[count] = 1
		for col in columns:
			if col != column:
				if correlations[col] > threshold:
					flags[tmp_count] = 1
					temp_cluster.append(col)
			tmp_count += 1
		clusters.append(temp_cluster)
		count += 1
	print("%s has/have %d clusters: " %(disk_model,len(clusters)))
	count = 1
	for cluster in clusters:
		print("cluster ",count,": ",cluster)
		count += 1

def feature_cluster(disk_model):
	startindex = -1
	is_ssd = None
	if disk_model.startswith('Disk'):
		startindex = 6
		is_ssd = False
	else:
		startindex = 3
		is_ssd = True
	file = open(os.path.join(path,disk_model,filename),'rb')
	failed_data = pickle.load(file)
	column = list(failed_data[0].columns[startindex:])
	sum = 0
	count = 0
	file = open(os.path.join(path,disk_model,datasetname),'rb')
	dataset = pickle.load(file)
	failed = dataset['failed']['X']
	dataset = failed
	drop_list = config_parse(disk_model,'drop_list')
	for record in dataset:
		record = pd.DataFrame(record,columns=column,index=[str(i) for i in range(len(record))])
		result = find_same(record)
		if judge(result,drop_list):
			record = record.drop(columns=drop_list) 
			corr = record.corr()
			corr = corr.fillna(0)
			if count == 0:
				sum = corr
			else:
				sum += corr
			count += 1
	sum /= count
	sum = sum.round(2)
	sum = sum.abs()
	column_len = len(sum.columns)
	map(sum,is_ssd)
	find_cluster(disk_model,sum)
	corr_plot(disk_model,sum)

def main():
	disk_model = 'Disk_1'
	feature_cluster(disk_model)

if __name__ == '__main__':
	main()