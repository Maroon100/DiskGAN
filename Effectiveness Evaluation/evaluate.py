import AdaSYN.adasyn as adasyn
import Mixup.mixup as mixup
import TimeGAN.timegan as timegan
import DiskGAN.diskgan as diskgan 
import Varify
import os
import pickle

datapath = '../data'
savepath = './'
disk_model = 'Disk_1'    #The selected disk model from Disk_1 to Disk_6
generative_model = 'diskgan'
genepath = './'
times = 1   # The ratio of generated failed sample size to that of original failed sample 
variation_times = 6 #The performance varies and need to be varifed for several times. We average the FDR,FAR and F1-Score in our result

def get_ori_num():
	file = open(os.path.join(datapath,disk_model,'samples.pkl'),'rb')
	data = pickle.load(file)
	num = sum([i[0] for i in data['Y']])
	return num

def main():
	ori_num = get_ori_num()
	gen_num = int(ori_num*times)
	if generative_model == 'adasyn':
		genename = disk_model+'_adasyn.pkl'
		adasyn.generate(disk_model,datapath,genepath,gen_num)
	elif generative_model == 'mixup':
		genename = disk_model+'_mixup.pkl'
		mixup.generate(disk_model,datapath,genepath,gen_num)
	elif generative_model == 'timegan':
		genename = disk_model+'_timegan.pkl'
		timegan.generate(disk_model,datapath,genepath,gen_num)
	elif generative_model == 'diskgan':
		genename = disk_model+'_diskgan.pkl'
		threshold = 0.3           # The threshold is used for our filter model to select high-qualified generated failed samples.
		filename = 'samples.pkl'  # The dataset used for training generative models. The samples.pkl contains the original failed samples. You can also choose to use part/extra data by setting the filename as training_1_16.pkl or training_2.pkl, where training_1_16 means only 1/16 data is used and training_2 means 2 times of original data is used.
		diskgan.generate(disk_model,datapath,filename,genepath,gen_num,threshold)
	else:                    # When generative_model is none, we use the original dataset to train a disk failure detection model
		genename = None

	for i in range(variation_times):
		Varify.varify(disk_model,datapath,genepath,genename)

if __name__ == '__main__':
	main()