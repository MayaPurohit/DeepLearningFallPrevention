# Maya Purohit
#4/19/2025
# Dataloader.py
# Develop a dataloader to get data models for the SRCNN

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import pandas as pd
from scipy.signal import find_peaks

# Allow loading of truncated images


class MotionDataset(Dataset):
    def __init__(self, root_dir, window_size, mode = "train", threshold = 3, test_ratio=0.2,seed=42, test_type = "normal"):

        self.root_dir = root_dir
        self.test_ratio = test_ratio
        self.seed = seed
        self.window_size = window_size
        self.mode = mode
        self.threshold = threshold
        

        random.seed(seed)
        np.random.seed(seed)

        if test_type == "normal":


            self.full_dataset = self.create_dataset_normal()
        
            indices = np.arange(len(self.full_dataset))
            np.random.shuffle(indices)

            total_len = len(indices)

            val_size = int((test_ratio/2)*total_len)
            train_size = total_len - 2*(val_size)


            train_ind = indices[:train_size]
            val_ind = indices[train_size:train_size + val_size]
            test_ind = indices[train_size + val_size:]

            self.test_data = self.full_dataset[test_ind]
            self.train_data = self.full_dataset[train_ind]
            self.val_data = self.full_dataset[val_ind]
        else:
            self.full_dataset, self.test_data = self.create_dataset_user()
        
            indices = np.arange(len(self.full_dataset))
            np.random.shuffle(indices)

            total_len = len(indices)

            val_size = int((test_ratio)*total_len)
            train_size = total_len - val_size


            train_ind = indices[:train_size]
            val_ind = indices[train_size:]

            test_indices =  np.arange(len(self.test_data))
            np.random.shuffle(test_indices)

            self.test_data = self.test_data[test_indices]
            self.train_data = self.full_dataset[train_ind]
            self.val_data = self.full_dataset[val_ind]



    def create_dataset_user(self):
        ActivityList = ['LocationA', 'LocationB', 'LocationC', 'LocationD', 'LocationE']
        PersonList = ['User1', 'User2', 'User3', 'User4', 'User5']

        index_to_remove = np.random.default_rng().integers(0, 5)
        
        removed_person = PersonList[index_to_remove]
        
        del PersonList[index_to_remove]
        dataset = []
        for i in range(len(PersonList)):
            for j in range(len(ActivityList)):
                data_pack = {}
                df = pd.read_csv(self.root_dir + fr"\\{PersonList[i]}_{ActivityList[j]}_Normal.csv", sep = "\t", header = 1)
                df = df[1:].astype('float64')
            
                df['Composed_Acceleration'] = np.sqrt(df['Shimmer_8665_Accel_LN_X_CAL']**2 + df['Shimmer_8665_Accel_LN_Y_CAL']**2 + df['Shimmer_8665_Accel_LN_Z_CAL']**2)
                # df['Composed_Gyroscope'] = np.sqrt(np.power(df['Shimmer_8665_Gyro_X_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Y_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Z_CAL'], 2))
                peaks, _ = find_peaks(df['Composed_Acceleration'], threshold = self.threshold, distance = self.window_size, prominence = 12 )

                for idx in peaks:
                    if (idx - ((self.window_size//2)-1)) > 0:
                        data_sample = df.iloc[idx - ((self.window_size//2)-1): idx + (self.window_size//2), 1:7]
                        data_pack["data_sample"] = torch.tensor(data_sample.to_numpy(), dtype=torch.float32)
                        data_pack["class_label"] = torch.tensor(j, dtype=torch.long)
                        dataset.append(data_pack)

        dataset = np.array(dataset)

        test_dataset = []
        for j in range(len(ActivityList)):
            data_pack = {}
            df = pd.read_csv(self.root_dir + fr"\\{removed_person}_{ActivityList[j]}_Normal.csv", sep = "\t", header = 1)
            df = df[1:].astype('float64')
        
            df['Composed_Acceleration'] = np.sqrt(df['Shimmer_8665_Accel_LN_X_CAL']**2 + df['Shimmer_8665_Accel_LN_Y_CAL']**2 + df['Shimmer_8665_Accel_LN_Z_CAL']**2)
            # df['Composed_Gyroscope'] = np.sqrt(np.power(df['Shimmer_8665_Gyro_X_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Y_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Z_CAL'], 2))
            peaks, _ = find_peaks(df['Composed_Acceleration'], threshold = self.threshold, distance = self.window_size, prominence = 12 )

            for idx in peaks:
                if (idx - ((self.window_size//2)-1)) > 0:
                    data_sample = df.iloc[idx - ((self.window_size//2)-1): idx + (self.window_size//2), 1:7]
                    data_pack["data_sample"] = torch.tensor(data_sample.to_numpy(), dtype=torch.float32)
                    data_pack["class_label"] = torch.tensor(j, dtype=torch.long)
                    test_dataset.append(data_pack)

        test_dataset = np.array(test_dataset)


        return dataset, test_dataset

    def create_dataset_normal(self):
        ActivityList = ['LocationA', 'LocationB', 'LocationC', 'LocationD', 'LocationE']
        PersonList = ['User1', 'User2', 'User3', 'User4', 'User5']
        dataset = []
        for i in range(len(PersonList)):
            for j in range(len(ActivityList)):
                data_pack = {}
                df = df = pd.read_csv(self.root_dir + fr"\\{PersonList[i]}_{ActivityList[j]}_Normal.csv", sep = "\t", header = 1)
                df = df[1:].astype('float64')
            
                df['Composed_Acceleration'] = np.sqrt(df['Shimmer_8665_Accel_LN_X_CAL']**2 + df['Shimmer_8665_Accel_LN_Y_CAL']**2 + df['Shimmer_8665_Accel_LN_Z_CAL']**2)
                # df['Composed_Gyroscope'] = np.sqrt(np.power(df['Shimmer_8665_Gyro_X_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Y_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Z_CAL'], 2))
                peaks, _ = find_peaks(df['Composed_Acceleration'], threshold = self.threshold, distance = self.window_size, prominence = 12 )

                for idx in peaks:
                    if (idx - ((self.window_size//2)-1)) > 0:
                        data_sample = df.iloc[idx - ((self.window_size//2)-1): idx + (self.window_size//2), 1:7]
                        data_pack["data_sample"] = torch.tensor(data_sample.to_numpy(), dtype=torch.float32)
                        data_pack["class_label"] = torch.tensor(j, dtype=torch.long)
                        dataset.append(data_pack)

        dataset = np.array(dataset)

        return dataset


    def __len__(self):
        if self.mode == "train":
            return len(self.train_data)
        elif self.mode == "val":
             return len(self.val_data)
        elif self.mode == "test":
             return len(self.test_data)

    

    def __getitem__(self, idx):
        if self.mode == "train":
            val = self.train_data[idx]
        elif self.mode == "val":
            val = self.val_data[idx]
        elif self.mode == "test":
            val = self.test_data[idx]
        return val

    
 

