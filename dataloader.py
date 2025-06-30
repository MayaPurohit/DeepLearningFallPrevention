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
from sklearn.preprocessing import MinMaxScaler



# Allow loading of truncated images


class MotionDataset(Dataset):
    def __init__(self, root_dir, window_size, mode = "train", threshold = 3, test_ratio=0.2, 
                 val_ratio = 0.2, normalize = False, input_channels = 6, user_num = 0, seed=42, test_type = "normal"):

        self.root_dir = root_dir
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.window_size = window_size
        self.mode = mode
        self.threshold = threshold
        self.index_to_remove_test = 1
        self.index_to_remove_val = 2
        self.normalize = normalize
        self.input_channels = input_channels
        self.user_num = user_num
        self.scaler = MinMaxScaler()

        random.seed(seed)
        np.random.seed(seed)

        if test_type == "normal":


            self.full_dataset = self.create_dataset_normal()
        
            indices = np.arange(len(self.full_dataset))
            np.random.shuffle(indices)

            total_len = len(indices)


            
            val_size = int(self.val_ratio * total_len)
            test_size = int(self.test_ratio * total_len)
            train_size = total_len - (val_size + test_size)


            train_ind = indices[:train_size]
            val_ind = indices[train_size:train_size + val_size]
            test_ind = indices[train_size + val_size:]

            self.test_data = self.full_dataset[test_ind]
            self.train_data = self.full_dataset[train_ind]
            self.val_data = self.full_dataset[val_ind]

        elif test_type == "user":
            self.full_dataset, self.test_data = self.create_dataset_user()
        
            indices = np.arange(len(self.full_dataset))
            np.random.shuffle(indices)

            total_len = len(indices)

            val_size = int((self.val_ratio)*total_len)
            train_size = total_len - val_size


            train_ind = indices[:train_size]
            val_ind = indices[train_size:]

            test_indices =  np.arange(len(self.test_data))
            np.random.shuffle(test_indices)

            self.test_data = self.test_data[test_indices]
            self.train_data = self.full_dataset[train_ind]
            self.val_data = self.full_dataset[val_ind]
        elif test_type == 'both':
            self.train_data, self.test_data, self.val_data = self.create_dataset_both()
        elif test_type == "individual":
            self.full_dataset = self.create_dataset_individual()
            val_size = int(self.val_ratio * total_len)
            test_size = int(self.test_ratio * total_len)
            train_size = total_len - (val_size + test_size)


            train_ind = indices[:train_size]
            val_ind = indices[train_size:train_size + val_size]
            test_ind = indices[train_size + val_size:]

            self.test_data = self.full_dataset[test_ind]
            self.train_data = self.full_dataset[train_ind]
            self.val_data = self.full_dataset[val_ind]







    def create_dataset_user(self):
        ActivityList = ['LocationA', 'LocationB', 'LocationC', 'LocationD', 'LocationE']
        PersonList = ['User1', 'User2', 'User3', 'User4', 'User5']

        
        
        removed_person = PersonList[self.index_to_remove_test]
        
        del PersonList[self.index_to_remove_test]
        dataset = []
        for i in range(len(PersonList)):
            for j in range(len(ActivityList)):
                data_pack = {}
                df = pd.read_csv(self.root_dir + fr"\\{PersonList[i]}_{ActivityList[j]}_Normal.csv", sep = "\t", header = 1)
                df = df[1:].astype('float64')
            
                df['Composed_Acceleration'] = np.sqrt(df['Shimmer_8665_Accel_LN_X_CAL']**2 + df['Shimmer_8665_Accel_LN_Y_CAL']**2 + df['Shimmer_8665_Accel_LN_Z_CAL']**2)
                df['Composed_Gyroscope'] = np.sqrt(np.power(df['Shimmer_8665_Gyro_X_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Y_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Z_CAL'], 2))
                peaks, _ = find_peaks(df['Composed_Acceleration'], threshold = self.threshold, distance = self.window_size, prominence = 12 )

                for idx in peaks:
                    if (idx - ((self.window_size//2)-1)) > 0:
                        data_sample = df.iloc[idx - ((self.window_size//2)-1): idx + (self.window_size//2), 1:7]
                        if self.normalize:
                            data_sample[:] = self.scaler.fit_transform(data_sample)
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
            df['Composed_Gyroscope'] = np.sqrt(np.power(df['Shimmer_8665_Gyro_X_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Y_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Z_CAL'], 2))
            peaks, _ = find_peaks(df['Composed_Acceleration'], threshold = self.threshold, distance = self.window_size, prominence = 12 )

            for idx in peaks:
                if (idx - ((self.window_size//2)-1)) > 0:
                    if self.input_channels == 2:
                        data_sample = df.iloc[idx - ((self.window_size//2)-1): idx + (self.window_size//2), 11:]
                    else:
                        data_sample = df.iloc[idx - ((self.window_size//2)-1): idx + (self.window_size//2), 1:7]
                    if self.normalize:
                        data_sample[:] = self.scaler.fit_transform(data_sample)
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
                df['Composed_Gyroscope'] = np.sqrt(np.power(df['Shimmer_8665_Gyro_X_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Y_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Z_CAL'], 2))
                peaks, _ = find_peaks(df['Composed_Acceleration'], threshold = self.threshold, distance = self.window_size, prominence = 12 )

                for idx in peaks:
                    if (idx - ((self.window_size//2)-1)) > 0:
                        if self.input_channels == 2:
                            data_sample = df.iloc[idx - ((self.window_size//2)-1): idx + (self.window_size//2), 11:]
                        else:
                            data_sample = df.iloc[idx - ((self.window_size//2)-1): idx + (self.window_size//2), 1:7]
                        if self.normalize:
                            data_sample[:] = self.scaler.fit_transform(data_sample)
                        data_pack["data_sample"] = torch.tensor(data_sample.to_numpy(), dtype=torch.float32)
                        data_pack["class_label"] = torch.tensor(j, dtype=torch.long)
                        dataset.append(data_pack)

        dataset = np.array(dataset)

        return dataset


    def create_dataset_both(self):
        ActivityList = ['LocationA', 'LocationB', 'LocationC', 'LocationD', 'LocationE']
        PersonList = ['User1', 'User2', 'User3', 'User4', 'User5']

        
        
        removed_person_test = PersonList[self.index_to_remove_test]
        
        removed_person_val = PersonList[self.index_to_remove_val]
        removed_people = [removed_person_test, removed_person_val]
        del PersonList[self.index_to_remove_test]
        del PersonList[self.index_to_remove_val]

        dataset = []
        for i in range(len(PersonList)):
            for j in range(len(ActivityList)):
                data_pack = {}
                df = pd.read_csv(self.root_dir + fr"\\{PersonList[i]}_{ActivityList[j]}_Normal.csv", sep = "\t", header = 1)
                df = df[1:].astype('float64')
            
                df['Composed_Acceleration'] = np.sqrt(df['Shimmer_8665_Accel_LN_X_CAL']**2 + df['Shimmer_8665_Accel_LN_Y_CAL']**2 + df['Shimmer_8665_Accel_LN_Z_CAL']**2)
                df['Composed_Gyroscope'] = np.sqrt(np.power(df['Shimmer_8665_Gyro_X_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Y_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Z_CAL'], 2))
                peaks, _ = find_peaks(df['Composed_Acceleration'], threshold = self.threshold, distance = self.window_size, prominence = 12 )

                for idx in peaks:
                    if (idx - ((self.window_size//2)-1)) > 0:
                        if self.input_channels == 2:
                            data_sample = df.iloc[idx - ((self.window_size//2)-1): idx + (self.window_size//2), 11:]
                        else:
                            data_sample = df.iloc[idx - ((self.window_size//2)-1): idx + (self.window_size//2), 1:7]
                        if self.normalize:
                            data_sample[:] = self.scaler.fit_transform(data_sample)
                        data_pack["data_sample"] = torch.tensor(data_sample.to_numpy(), dtype=torch.float32)
                        data_pack["class_label"] = torch.tensor(j, dtype=torch.long)
                        dataset.append(data_pack)

        dataset = np.array(dataset)

        test_val_set = [[], []]
        for i in range(len(removed_people)):
            for j in range(len(ActivityList)):
                data_pack = {}
                df = pd.read_csv(self.root_dir + fr"\\{removed_people[i]}_{ActivityList[j]}_Normal.csv", sep = "\t", header = 1)
                df = df[1:].astype('float64')
            
                df['Composed_Acceleration'] = np.sqrt(df['Shimmer_8665_Accel_LN_X_CAL']**2 + df['Shimmer_8665_Accel_LN_Y_CAL']**2 + df['Shimmer_8665_Accel_LN_Z_CAL']**2)
                df['Composed_Gyroscope'] = np.sqrt(np.power(df['Shimmer_8665_Gyro_X_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Y_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Z_CAL'], 2))
                peaks, _ = find_peaks(df['Composed_Acceleration'], threshold = self.threshold, distance = self.window_size, prominence = 12 )

                for idx in peaks:
                    if (idx - ((self.window_size//2)-1)) > 0:
                        if self.input_channels == 2:
                            data_sample = df.iloc[idx - ((self.window_size//2)-1): idx + (self.window_size//2), 11:]
                        else:
                            data_sample = df.iloc[idx - ((self.window_size//2)-1): idx + (self.window_size//2), 1:7]
                        if self.normalize:
                            data_sample[:] = self.scaler.fit_transform(data_sample)
                        data_pack["data_sample"] = torch.tensor(data_sample.to_numpy(), dtype=torch.float32)
                        data_pack["class_label"] = torch.tensor(j, dtype=torch.long)
                        test_val_set[i].append(data_pack)

            test_val_set[i] = np.array(test_val_set[i])


        return dataset, test_val_set[0], test_val_set[1]
    
    def create_dataset_individual(self):
        ActivityList = ['LocationA', 'LocationB', 'LocationC', 'LocationD', 'LocationE']
        PersonList = ['User1', 'User2', 'User3', 'User4', 'User5']

        chosen_ind = PersonList[self.user_num]
        dataset = []
        for j in range(len(ActivityList)):
            data_pack = {}
            df = df = pd.read_csv(self.root_dir + fr"\\{chosen_ind}_{ActivityList[j]}_Normal.csv", sep = "\t", header = 1)
            df = df[1:].astype('float64')
        
            df['Composed_Acceleration'] = np.sqrt(df['Shimmer_8665_Accel_LN_X_CAL']**2 + df['Shimmer_8665_Accel_LN_Y_CAL']**2 + df['Shimmer_8665_Accel_LN_Z_CAL']**2)
            df['Composed_Gyroscope'] = np.sqrt(np.power(df['Shimmer_8665_Gyro_X_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Y_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Z_CAL'], 2))
            peaks, _ = find_peaks(df['Composed_Acceleration'], threshold = self.threshold, distance = self.window_size, prominence = 12 )

            for idx in peaks:
                if (idx - ((self.window_size//2)-1)) > 0:
                    if self.input_channels == 2:
                        data_sample = df.iloc[idx - ((self.window_size//2)-1): idx + (self.window_size//2), 11:]
                    else:
                        data_sample = df.iloc[idx - ((self.window_size//2)-1): idx + (self.window_size//2), 1:7]
                    if self.normalize:
                        data_sample[:] = self.scaler.fit_transform(data_sample)
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

    
 

