import pickle
import os
import numpy as np 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from keras.models import load_model
import tensorflow as tf

path = './'
trainfile = 'train.pkl'
testfile = 'test.pkl'
modelfile = 'trained_model.h5'

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

def split(dataset):
	failed = []
	count = 0
	records = dataset['X'].tolist()
	labels = dataset['Y'].tolist()
	for record in records:
		if labels[count][0] == 1:
			failed.append(record)
		count += 1
	print(count)
	return np.asarray(failed)

def plot(train,wrong):
	fontsize = 10
	data = np.concatenate((train,wrong))
	label = np.array(['train']*train.shape[0]+['wrong']*wrong.shape[0])

	x_tsne = TSNE(n_components=2,random_state=32).fit_transform(data,label)
	fig,ax = plt.subplots(figsize=(5,3))
	wrong = ax.scatter(x_tsne[(label=='wrong'),0],x_tsne[(label=='wrong'),1],label='wrong',c='r',marker='x')
	train = ax.scatter(x_tsne[(label=='train'),0],x_tsne[(label=='train'),1],label='train',c='g',marker='+')
	ax.legend((train, wrong), ("Train", "Wrong"), loc = "best",fontsize=fontsize)
	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)
	myfig = plt.gcf()
	plt.show()
	myfig.savefig(path+'/explore.svg', format='svg')
	print('The generated figure has been saved !!!')

def main():
	file = open(os.path.join(path,trainfile),'rb')
	train_failure = split(pickle.load(file))
	file = open(os.path.join(path,testfile),'rb')
	test_failure = split(pickle.load(file))
	model = load_model(os.path.join(path, modelfile),
                       custom_objects={"metric_F1score": metric_F1score, "metric_FDR": metric_FDR,
                                       "metric_FAR": metric_FAR})
	predictions = model.predict(test_failure)
	count = 0
	right = []
	wrong = []
	for predict in predictions:
		if predict[0]>=0.5:
			right.append(test_failure[count])
		else:
			wrong.append(test_failure[count])
		count += 1
	right = np.asarray(right)
	wrong = np.asarray(wrong)
	train_failure = train_failure.reshape((train_failure.shape[0],-1))
	right = right.reshape((right.shape[0],-1))
	wrong = wrong.reshape((wrong.shape[0],-1))
	plot(train_failure,wrong)

if __name__ == '__main__':
	main()