import pickle
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from math import ceil
import torch.nn as nn

import DiskGAN.discriminator as discriminator
import DiskGAN.generator as generator
import DiskGAN.helpers as helpers
import DiskGAN.generate_samples as generate_samples
import configparser

CUDA = False
VOCAB_SIZE = 1000

MAX_SEQ_LEN = 6
START_LETTER = 0
BATCH_SIZE = 32
MLE_TRAIN_EPOCHS = 100
ADV_TRAIN_EPOCHS = 100

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 32

validation_num = -1
R_POS_NEG_SAMPLES = -1
POS_NEG_SAMPLES = -1

SMART_ATTRIBUTE = 999       #The size of action space

def config_parse(section,key):
	cf = configparser.ConfigParser()
	cf.read('../configuration.ini')
	value = cf.get(section,key).split(',')
	return value

def load_data(path,id):
    file = open(path,'rb')
    records = pickle.load(file)
    labels = records['Y']
    records = records['X']
    sequences = []
    sample_len = len(records[0])
    features_len = len(records[0][0])
    count = 0
    for record in records:
        if labels[count] == 1:
            record = pd.DataFrame(record,columns=[str(i) for i in range(features_len)],index=[i for i in range(sample_len)])
            temp = [int(i*SMART_ATTRIBUTE) for i in record[str(id)]]
            temp = np.asarray(temp)
            sequences.append(temp)
        count += 1
    return sequences

def train_generator_MLE(gen, gen_opt, real_data_samples, epochs):

    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0

        for i in range(0, POS_NEG_SAMPLES, BATCH_SIZE):
            inp, target = helpers.prepare_generator_batch(real_data_samples[i:i + BATCH_SIZE],
                                                          start_letter=START_LETTER,
                                                          gpu=CUDA)
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(inp, target)
            loss.backward()
            gen_opt.step()

            total_loss += loss.data.item()

            if (i / BATCH_SIZE) % ceil(
                            ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                print('.', end='')
                sys.stdout.flush()

            # each loss in a batch is loss per sample
            total_loss = total_loss / ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / MAX_SEQ_LEN


            print(' average_train_NLL = %.4f' % (total_loss))


def train_generator_PG(gen, gen_opt,dis, num_batches):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """

    for batch in range(num_batches):
        s = gen.sample(BATCH_SIZE*2)        # 64 works best
        inp, target = helpers.prepare_generator_batch(s, start_letter=START_LETTER, gpu=CUDA)
        rewards = dis.batchClassify(target)

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards)
        pg_loss.backward()
        gen_opt.step()

def train_discriminator(discriminator, dis_opt, real_data_samples, generator, d_steps, epochs):
    pos_val = real_data_samples[:validation_num]
    neg_val = generator.sample(validation_num)
    real_data_samples = real_data_samples[validation_num:]

    val_inp, val_target = helpers.prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)

    for d_step in range(d_steps):
        s = helpers.batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
        dis_inp, dis_target = helpers.prepare_discriminator_data(real_data_samples, s, gpu=CUDA)

        for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0, 2 * POS_NEG_SAMPLES, BATCH_SIZE):
                inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out > 0.5) == (target > 0.5)).data.item()

                if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                        BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
            total_acc /= float(2 * POS_NEG_SAMPLES)

            val_pred = discriminator.batchClassify(val_inp)
            print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                total_loss, total_acc, torch.sum((val_pred > 0.5) == (val_target > 0.5)).data.item() / (2*validation_num)))

def feature_sample_gene(disk_model,path,filename,savepath,gen_num,feature):
    positive_samples = load_data(os.path.join(path,disk_model,filename),id=feature)
    global R_POS_NEG_SAMPLES,validation_num,POS_NEG_SAMPLES
    R_POS_NEG_SAMPLES = len(positive_samples)
    validation_num = int(R_POS_NEG_SAMPLES*0.2)    # Here, we split the training set and test set with ratio 8:2 by default
    POS_NEG_SAMPLES = R_POS_NEG_SAMPLES - validation_num

    positive_samples = np.asarray(positive_samples)

    positive_samples = torch.from_numpy(positive_samples).type(dtype=torch.int)

    perm = torch.randperm(R_POS_NEG_SAMPLES)

    positive_samples = positive_samples[perm]

    gen = generator.Generator(GEN_EMBEDDING_DIM,GEN_HIDDEN_DIM,VOCAB_SIZE,MAX_SEQ_LEN)
    dis = discriminator.Discriminator(DIS_EMBEDDING_DIM,DIS_HIDDEN_DIM,VOCAB_SIZE,MAX_SEQ_LEN,CUDA)

    print('Starting Generator MLE Training...')
    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)
    train_generator_MLE(gen, gen_optimizer, positive_samples, MLE_TRAIN_EPOCHS)

    print('\nStarting Discriminator Training...')

    dis_optimizer = optim.Adam(dis.parameters(),lr=1e-2)
    train_discriminator(dis, dis_optimizer, positive_samples, gen, 50, 3)  # The discriminator has to been pre-trained for 50 epochs

    print('\nStarting Adversarial Training...')
    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch + 1))

        print('\nAdversarial Training Generator : ', end='')
        sys.stdout.flush()
        train_generator_PG(gen, gen_optimizer,dis, 1)

        print('\nAdversarial Training Discriminator : ')
        train_discriminator(dis, dis_optimizer, positive_samples, gen,10, 5)
    
    gene_samples = gen.sample(gen_num).numpy()
    file = open('./gene_samples_'+str(feature)+'.pkl','wb')
    pickle.dump(gene_samples,file)
    print("Done!!!")

def generate(disk_model,datapath,filename,savepath,gen_num,threshold):
    feature_num = 50                      # The generated sample size of base features
    if disk_model == 'Disk_6':
        feature_num = 200
    base_features = [int(i) for i in config_parse(disk_model,'base_features')]
    for feature in base_features:
        feature_sample_gene(disk_model,datapath,filename,savepath,feature_num,feature)
    generate_samples.gen(disk_model,datapath,savepath,'./DiskGAN','./',gen_num,threshold)

if __name__ == '__main__':
	generate()