import pickle
import os
import numpy as np
import random

filename = 'samples.pkl'
alpha = 0.5

def split(dataset):
	failed = []
	healthy = []
	count = 0
	records = dataset['X'].tolist()
	labels = dataset['Y'].tolist()
	for record in records:
		if labels[count][0] == 1:
			failed.append(record)
		else:
			healthy.append(record)
		count += 1
	return failed,healthy

def generate(disk_model,path,savepath,gen_num):
	file = open(os.path.join(path,disk_model,filename),'rb')
	dataset = pickle.load(file)
	failed,healthy = split(dataset)
	generated = []
	labels = []
	heal_len = len(healthy)
	fail_len = len(failed)
	gene_len = len(failed)
	for i in range(gene_len):
		seed = random.randint(0,fail_len-1)
		sample = failed[seed]
		index = random.randint(0,heal_len-1)
		temp_sample = np.asarray(healthy[index])
		lam = np.random.beta(alpha,alpha)
		sample = np.asarray(sample)
		x = lam*sample + (1-lam)*temp_sample
		y = lam
		generated.append(x.tolist())
		labels.append([y])
	generated = generated[:gen_num]
	labels = labels[:gen_num]
	generated_data = {'X':generated,'Y':labels}
	file = open(os.path.join(savepath,disk_model+'_mixup.pkl'),'wb')
	pickle.dump(generated_data,file)
	print('Generated samples has been successfully saved!!!')