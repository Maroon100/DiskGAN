import numpy as np
import torch
import pickle

def normalize(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def to_tensor(data):
    return torch.from_numpy(data).float()


def batch_generator(dataset, batch_size):
    dataset_size = len(dataset)
    idx = torch.randperm(dataset_size)
    batch_idx = idx[:batch_size]
    batch = torch.stack([to_tensor(dataset[i]) for i in batch_idx])
    return batch

class TimeDataset(torch.utils.data.Dataset):
    def __init__(self,data_path):
        data = pickle.load(open(data_path,'rb'))
        failed = self.split(data)
        self.samples = []
        idx = torch.randperm(len(failed))
        for i in range(len(failed)):
            self.samples.append(failed[idx[i]])
 
    def __getitem__(self,index):
        return self.samples[index]
    
    def __len__(self):
        return len(self.samples)

    def split(self,data):
        failed = []
        count = 0
        records = data['X'].tolist()
        labels = data['Y'].tolist()
        for record in records:
            if labels[count][0] == 1:
                failed.append(np.asarray(record))
            count += 1
        return failed