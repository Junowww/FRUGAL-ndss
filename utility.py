import numpy as np
import torch
import pickle
import sys
import os
import logging as log
# import dill
import logging

# set to True to enable debugging information
DEBUG_ON = True


def save_checkpoint(actor, subdir):
    checkpoint_dir = './checkpoint'
    checkpoint_path = os.path.join(checkpoint_dir, f'{subdir}_checkpoint_{iteration}.pth')
    checkpoint = {
        'actor_state_dict': actor.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved to {checkpoint_path}')


def load_checkpoint(actor, attack_model, subdir, iteration, device):
    checkpoint_dir = './checkpoint'
    checkpoint_path = os.path.join(checkpoint_dir,f'{subdir}_checkpoint_{iteration}.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    attack_model.load_state_dict(checkpoint['attack_model_state_dict'])
    # iteration = checkpoint['iteration']
    print(f'Checkpoint loaded from {checkpoint_path} at iteration {iteration}')
    return iteration

def LoadDataNoDefCW():
    print("Loading non-defended dataset for closed-world scenario")

    dataset_dir = './dataset/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding labels (websites' labels)

    # Load testing data
    with open(dataset_dir + 'X_test_NoDef.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle, encoding='iso-8859-1'))
    with open(dataset_dir + 'y_test_onehot.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle, encoding='iso-8859-1'))

    print("Data dimensions:")
    print("X: Testing data's shape: ", X_test.shape)
    print("y: Testing data's shape: ", y_test.shape)

    # Convert to PyTorch Tensors
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_test, y_test

def LoadGoodSampleCW(limits):
    print("Loading non-defended Dataset for closed-world scenario" )
    # Point to the directory storing data
    dataset_dir = './dataset/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load good sample
    with open(dataset_dir + 'CW_X_train_goodsamples_{}.pkl'.format(limits), 'rb') as handle:
        X_goodSample = np.array(pickle.load(handle,encoding='iso-8859-1'))
    with open(dataset_dir + 'CW_y_train_goodsamples_{}.pkl'.format(limits), 'rb') as handle:
        y_goodSample = np.array(pickle.load(handle,encoding='iso-8859-1'))

    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_goodSample.shape)
    print ("y: Training data's shape : ", y_goodSample.shape)

    X_goodSample = torch.tensor(X_goodSample, dtype=torch.float32)
    y_goodSample = torch.tensor(y_goodSample, dtype=torch.long)

    return X_goodSample, y_goodSample

def LoadGoodSampleTestCW():
    print("Loading non-defended Dataset for closed-world scenario" )
    # Point to the directory storing data
    dataset_dir = '.'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load good sample
    with open(dataset_dir + 'CW_X_test_goodsamples_20.pkl', 'rb') as handle:
        X_goodSample = np.array(pickle.load(handle,encoding='iso-8859-1'))
    with open(dataset_dir + 'CW_y_test_goodsamples_20.pkl', 'rb') as handle:
        y_goodSample = np.array(pickle.load(handle,encoding='iso-8859-1'))

    print ("Data dimensions:")
    print ("X: Testing data's shape : ", X_goodSample.shape)
    print ("y: Testing data's shape : ", y_goodSample.shape)

    X_goodSample = torch.tensor(X_goodSample, dtype=torch.float32)
    y_goodSample = torch.tensor(y_goodSample, dtype=torch.long)

    return X_goodSample, y_goodSample

class MyDataset():
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx])  
        y = torch.LongTensor(self.labels[idx])  
        return x, y


class WholeDatasetIterator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.dataset):
            item = self.dataset[self.index]
            self.index += 1
            return item
        else:
            raise StopIteration

