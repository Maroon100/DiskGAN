import pickle
import os
import numpy as np
import math
import random

filename = 'samples.pkl'
K = 5

def split(dataset):
	failed = []
	indexes = []
	count = 0
	records = dataset['X'].tolist()
	labels = dataset['Y'].tolist()
	for record in records:
		if labels[count][0] == 1:
			failed.append(record)
			indexes.append(count)
		count += 1
	return failed,indexes

def cosine_similarity(x,y):
	num = x.dot(y.T)
	denom = np.linalg.norm(x)*np.linalg.norm(y)
	return num/denom

def get_neighbours(f_sample,K,samples):
	distances = {}
	sample_len = len(f_sample)
	index = 0
	for data in samples:
		summary = 0
		for i in range(sample_len):
			summary += cosine_similarity(np.array(f_sample[i]),np.array(data[i]))
		summary /= sample_len
		distances[str(index)] = summary
		index += 1
	sorted_distance = sorted(distances.items(),key=lambda item:item[1],reverse=True)
	target = []
	for i in range(1,K+1):
		target.append(int(sorted_distance[i][0]))
	return target

def get_info(count,knn_samples,indexes):
	info = {'healthy_sample':0,'failed_neibour':[],'seed_num':0}
	for i in knn_samples:
		if i in indexes:
			info['failed_neibour'].append(i)
		else:
			info['healthy_sample'] += 1
	return info

def gen(sample,neibour):
	sample_len = len(sample)
	attri_num = len(sample[0])
	random_seeds = [random.random() for i in range(attri_num)]
	synthetic_sample = []
	for i in range(sample_len):
		data = sample[i]
		synthetic_data = []
		for j in range(attri_num):
			temp = data[j] + random_seeds[j]*(neibour[i][j]-data[j])
			synthetic_data.append(temp)
		synthetic_sample.append(synthetic_data)
	synthetic_sample = np.array(synthetic_sample)
	return synthetic_sample

def generate(disk_model,path,savepath,gen_num):
	file = open(os.path.join(path,disk_model,filename),'rb')
	dataset = pickle.load(file)
	failed,indexes = split(dataset)
	gen_samples = len(failed)
	total = dataset['X'].tolist()
	count = 0
	records = {}
	summary = 0
	generated = []
	for f_sample in failed:
		knn_samples = get_neighbours(f_sample,K,total)
		info = get_info(count,knn_samples,indexes)
		if info['healthy_sample']>0 and len(info['failed_neibour'])>0:
			records[str(count)] = info
			summary += info['healthy_sample']
		count += 1

	for key,value in records.items():
		records[key]['seed_num'] = math.ceil(value['healthy_sample']/summary*gen_num)
		for i in range(value['seed_num']):
			random_num = random.randint(0,len(records[key]['failed_neibour'])-1)
			temp = gen(total[int(key)],total[value['failed_neibour'][random_num]])
			generated.append(temp.tolist())

	generated = generated[:gen_num]
	labels = [[1]]*gen_num
	generated_data = {'X':generated,'Y':labels}
	file = open(os.path.join(savepath,disk_model+'_adasyn.pkl'),'wb')
	pickle.dump(generated_data,file)
	print('Generated samples has been successfully saved!!!')