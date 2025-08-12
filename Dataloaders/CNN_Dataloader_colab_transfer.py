# Maya Purohit
#6/19/2025
# CNN_Dataloader_colab_transfer.py
# Develop a dataloader to train CNN models for surface detection using transfer learning mechanism 

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
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import svm




'''Types of Data Setups
    # create_dataset_user: create training set with 4 users and create testing set with the remaining user 
    # create_dataset_normal: create training, validation, and testing set with all 5 users depending on the ratio defined above
    # create_dataset_individual: create training, validation, and testing set with data from one individual user with the ratio and user defined above
    # create_dataset_olivia and create_dataset_maya: read in data that is generated from txt or npy files 
'''


class MotionDataset(Dataset):
    def __init__(self, root_dir, window_size, mode = "train", test_ratio=0.2, 
                 val_ratio = 0.2, normalize = False, input_channels = 6, user_num = 1, seed=42, num_stacks = 3, test_type = "normal"):

        self.root_dir = root_dir
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.window_size = window_size
        self.mode = mode
        self.index_to_remove_test = 1
        self.index_to_remove_val = 2
        self.normalize = normalize
        self.input_channels = input_channels
        self.user_num = user_num
        self.num_stacks = num_stacks
        self.scaler = MinMaxScaler()

        random.seed(seed)
        np.random.seed(seed)


        #Create general four user and one user dataset
        self.four_users, self.one_user = self.create_dataset_user()
        self.four_separate_indices = np.arange(len(self.four_users))
        np.random.shuffle(self.four_separate_indices)

    
        four_total_len = len(self.four_separate_indices)

        # define ratios
        four_val_size = int((self.val_ratio)*four_total_len)
        four_test_size = int(self.test_ratio * four_total_len)
        four_train_size = four_total_len - (four_val_size + four_test_size)

        self.one_separate_indices = np.arange(len(self.one_user))
        np.random.shuffle(self.one_separate_indices)

        one_total_len = len(self.one_separate_indices)


        # Set up sizes
        one_val_size = int(self.val_ratio * one_total_len)
        one_test_size = int(self.test_ratio * one_total_len)
        one_train_size = one_total_len - (one_val_size + one_test_size)
        self.test_ind = self.one_separate_indices[one_train_size + one_val_size:]

    

        if test_type == "normal" or test_type == "olivia" or test_type == "maya":


            #Made the dataset based on the type of test being run 
            self.all_labels = []
            if test_type == "normal":
                self.full_dataset = self.create_dataset_normal()
            elif test_type == "olivia":
                self.full_dataset = self.create_dataset_olivia()
            elif test_type == "maya":
                self.full_dataset = self.create_dataset_maya()
        
            indices = np.arange(len(self.full_dataset))
            np.random.shuffle(indices)



            total_len = len(indices)


            # Set up sizes
            val_size = int(self.val_ratio * total_len)
            test_size = int(self.test_ratio * total_len)
            train_size = total_len - (val_size + test_size)



            #Make datasets based on their indices 
            train_ind = indices[:train_size]
            val_ind = indices[train_size:train_size + val_size]
            test_ind = indices[train_size + val_size:]


            self.test_data = self.full_dataset[test_ind]
            self.train_data = self.full_dataset[train_ind]
            self.val_data = self.full_dataset[val_ind]


        elif test_type == "individual":



            #Make datasets based on their indices from the one user set 
            train_ind = self.one_separate_indices[:one_train_size]
            val_ind = self.one_separate_indices[one_train_size:one_train_size + one_val_size]


            #define the three sets
            self.test_data = self.one_user[self.test_ind]
            self.train_data = self.one_user[train_ind]
            self.val_data = self.one_user[val_ind]



        elif test_type == "user":


            #make training and valifation set from the four user dataset
            train_ind = self.four_separate_indices[:four_train_size]
            val_ind = self.four_separate_indices[four_train_size:four_train_size + four_val_size]


            #define test set from one user 
            self.test_data = self.one_user[self.one_separate_indices]
            self.train_data = self.four_users[train_ind]
            self.val_data = self.four_users[val_ind]

        elif test_type == "normal_user":

            # Four Users -- all data not included in training or validation
            four_test_ind = self.four_separate_indices[four_train_size + four_val_size: ]
            four_user_test = self.four_users[four_test_ind]



            #Final Individual User -- all data not included in training or validation
            individual_test = self.one_user[self.test_ind]

            #make only testing set 
            self.test_data = np.concatenate((four_user_test, individual_test))
            test_indices = np.arange(len(self.test_data))
            self.test_data = self.test_data[test_indices]









# Dataset with  4 users in train, 5th user in test 
    def create_dataset_user(self):
        ActivityList = ['LocationA', 'LocationB', 'LocationC', 'LocationD', 'LocationE']
        PersonList = ['User1', 'User2', 'User3', 'User4', 'User5']

        
        #Remove one person from the training set to test on 
        removed_person = PersonList[self.index_to_remove_test]
        
        del PersonList[self.index_to_remove_test]
        dataset = []
        for i in range(len(PersonList)):
            for j in range(len(ActivityList)):
                
                df = pd.read_csv(self.root_dir + fr"/{PersonList[i]}_{ActivityList[j]}_Normal.csv", sep = "\t", header = 1)
                df = df[1:].astype('float64')
            
                #Compose the acceleration and gyroscope data
                df['Composed_Acceleration'] = np.sqrt(df['Shimmer_8665_Accel_LN_X_CAL']**2 + df['Shimmer_8665_Accel_LN_Y_CAL']**2 + df['Shimmer_8665_Accel_LN_Z_CAL']**2)
                df['Composed_Gyroscope'] = np.sqrt(np.power(df['Shimmer_8665_Gyro_X_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Y_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Z_CAL'], 2))


                #Normalized composed data to find peaks 
                normalized_acceleration = (df['Composed_Acceleration'] - df['Composed_Acceleration'].mean())/ (df['Composed_Acceleration'].std())
                normalized_gyroscope = (df['Composed_Gyroscope'] - df['Composed_Gyroscope'].mean())/ (df['Composed_Gyroscope'].std())
                peaks, _ = find_peaks(normalized_acceleration, distance = self.window_size*2, prominence = 2)

                
                #Replace data with normalized data
                if self.normalize == True:
                    df['Composed_Acceleration'] = normalized_acceleration
                    df['Composed_Gyroscope'] = normalized_gyroscope


                for idx in peaks:
                    #Re-initialize the data pack everytime to ensure unique data samples 
                    data_pack = {}
                    if (idx - ((self.window_size//2)-1)) > 0:
                        if self.input_channels == 2:
                            data_sample = df.iloc[idx - ((self.window_size//2)): idx + (self.window_size//2), 11:]
                        else:
                            data_sample = df.iloc[idx - ((self.window_size//2)): idx + (self.window_size//2), 1:7]


                        #Stack data horizontally
                        data_sample = data_sample.to_numpy()
                        columns_to_copy = data_sample.copy()
                        for m in range(self.num_stacks -1):
                            data_sample = np.hstack([data_sample, columns_to_copy])
                        #Set data and label in data 
                        data_pack["data_sample"] = torch.tensor(data_sample, dtype=torch.float32)
                        data_pack["class_label"] = torch.tensor(j, dtype=torch.long)
                        dataset.append(data_pack)

        dataset = np.array(dataset)


        #Repeat for User that is not included in the training set 
        test_dataset = []
        for j in range(len(ActivityList)):
            
            df = pd.read_csv(self.root_dir + fr"/{removed_person}_{ActivityList[j]}_Normal.csv", sep = "\t", header = 1)
            df = df[1:].astype('float64')
        
            df['Composed_Acceleration'] = np.sqrt(df['Shimmer_8665_Accel_LN_X_CAL']**2 + df['Shimmer_8665_Accel_LN_Y_CAL']**2 + df['Shimmer_8665_Accel_LN_Z_CAL']**2)
            df['Composed_Gyroscope'] = np.sqrt(np.power(df['Shimmer_8665_Gyro_X_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Y_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Z_CAL'], 2))
            
            normalized_acceleration = (df['Composed_Acceleration'] - df['Composed_Acceleration'].mean())/ (df['Composed_Acceleration'].std())
            normalized_gyroscope = (df['Composed_Gyroscope'] - df['Composed_Gyroscope'].mean())/ (df['Composed_Gyroscope'].std())
            peaks, _ = find_peaks(normalized_acceleration, distance = self.window_size*2, prominence = 2)

            #Replace data with normalized data
            if self.normalize == True:
                df['Composed_Acceleration'] = normalized_acceleration
                df['Composed_Gyroscope'] = normalized_gyroscope

            #make dataset from identified peaks 
            for idx in peaks:
                data_pack = {}
                if (idx - ((self.window_size//2)-1)) > 0:
                    if self.input_channels == 2:
                        data_sample = df.iloc[idx - ((self.window_size//2)): idx + (self.window_size//2), 11:]
                    else:
                        data_sample = df.iloc[idx - ((self.window_size//2)): idx + (self.window_size//2), 1:7]
                    

                    #Stack data horizontally
                    data_sample = data_sample.to_numpy()
                    columns_to_copy = data_sample.copy()
                    for m in range(self.num_stacks -1):
                        data_sample = np.hstack([data_sample, columns_to_copy])
                    data_pack["data_sample"] = torch.tensor(data_sample, dtype=torch.float32)
                    data_pack["class_label"] = torch.tensor(j, dtype=torch.long)
                    test_dataset.append(data_pack)

        test_dataset = np.array(test_dataset)


        return dataset, test_dataset

    # Dataset with all 5 users (5 users in train, val and test set)
    def create_dataset_normal(self):
        ActivityList = ['LocationA', 'LocationB', 'LocationC', 'LocationD', 'LocationE']
        PersonList = ['User1', 'User2', 'User3', 'User4', 'User5']
        dataset = []
        for i in range(len(PersonList)):
            for j in range(len(ActivityList)):
                #Iterate through all users and locations and extract samples into one large dataset 
                df = pd.read_csv(self.root_dir + fr"/{PersonList[i]}_{ActivityList[j]}_Normal.csv", sep = "\t", header = 1)
                df = df[1:].astype('float64')
            
                #Compose the acceleration and gyroscope data
                df['Composed_Acceleration'] = np.sqrt(df['Shimmer_8665_Accel_LN_X_CAL']**2 + df['Shimmer_8665_Accel_LN_Y_CAL']**2 + df['Shimmer_8665_Accel_LN_Z_CAL']**2)
                df['Composed_Gyroscope'] = np.sqrt(np.power(df['Shimmer_8665_Gyro_X_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Y_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Z_CAL'], 2))


                #normalize composed data
                normalized_acceleration = (df['Composed_Acceleration'] - df['Composed_Acceleration'].mean())/ (df['Composed_Acceleration'].std())
                normalized_gyroscope = (df['Composed_Gyroscope'] - df['Composed_Gyroscope'].mean())/ (df['Composed_Gyroscope'].std())
                peaks, _ = find_peaks(normalized_acceleration, distance = self.window_size*2, prominence = 2)

                #if normalize, then replace raw data with normalized data
                if self.normalize == True:
                    df['Composed_Acceleration'] = normalized_acceleration
                    df['Composed_Gyroscope'] = normalized_gyroscope
                
                for idx in peaks:
                    data_pack = {}
                    if (idx - ((self.window_size//2)-1)) > 0:
                        if self.input_channels == 2:
                            data_sample = df.iloc[idx - ((self.window_size//2)): idx + (self.window_size//2), 11:]
                            #np.savetxt(f"C:\\Users\\mayam\\DeepLearningFallDetection\\SavedData\\User1_{ActivityList[j]}_Normal_{i+1}.dat", data_sample, delimiter=',')
                        else:
                            data_sample = df.iloc[idx - ((self.window_size//2)): idx + (self.window_size//2), 1:7]

                        #Stack data horizontally
                        data_sample = data_sample.to_numpy()
                        columns_to_copy = data_sample.copy()
                        for m in range(self.num_stacks -1):
                            data_sample = np.hstack([data_sample, columns_to_copy])
                        
                        data_pack["class_label"] = torch.tensor(j, dtype=torch.long)
                        data_pack["data_sample"] = torch.tensor(data_sample, dtype=torch.float32)

                        
                        dataset.append(data_pack)

        dataset = np.array(dataset)

        return dataset

    
    # Dataset for only one user 
    def create_dataset_individual(self):
        ActivityList = ['LocationA', 'LocationB', 'LocationC', 'LocationD', 'LocationE']
        PersonList = ['User1', 'User2', 'User3', 'User4', 'User5']

        #Make dataset for one individual user 
        chosen_ind = PersonList[self.user_num - 1]
        dataset = []
        for j in range(len(ActivityList)):
            
            df = pd.read_csv(self.root_dir + fr"/{chosen_ind}_{ActivityList[j]}_Normal.csv", sep = "\t", header = 1)
            df = df[1:].astype('float64')
        
            #Compose data and acceleration 
            df['Composed_Acceleration'] = np.sqrt(df['Shimmer_8665_Accel_LN_X_CAL']**2 + df['Shimmer_8665_Accel_LN_Y_CAL']**2 + df['Shimmer_8665_Accel_LN_Z_CAL']**2)
            df['Composed_Gyroscope'] = np.sqrt(np.power(df['Shimmer_8665_Gyro_X_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Y_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Z_CAL'], 2))
            
            #Normalize the data 
            normalized_acceleration = (df['Composed_Acceleration'] - df['Composed_Acceleration'].mean())/ (df['Composed_Acceleration'].std())
            normalized_gyroscope = (df['Composed_Gyroscope'] - df['Composed_Gyroscope'].mean())/ (df['Composed_Gyroscope'].std())
            peaks, _ = find_peaks(normalized_acceleration, distance = self.window_size*2, prominence = 2)

            #if it should be normalized, replace the raw composition with the normalized data
            if self.normalize == True:
                df['Composed_Acceleration'] = normalized_acceleration
                df['Composed_Gyroscope'] = normalized_gyroscope
            
            #create data samples around detected peaks of size window size
            for idx in peaks:
                data_pack = {}
                if (idx - ((self.window_size//2)-1)) > 0:
                    if self.input_channels == 2:
                        data_sample = df.iloc[idx - ((self.window_size//2)): idx + (self.window_size//2), 11:]
                        #np.savetxt(f"C:\\Users\\mayam\\DeepLearningFallDetection\\SavedData\\User1_{ActivityList[j]}_Normal_{i+1}.dat", data_sample, delimiter=',')
                    else:
                        data_sample = df.iloc[idx - ((self.window_size//2)): idx + (self.window_size//2), 1:7]

                    #Stack data 
                    data_sample = data_sample.to_numpy()
                    columns_to_copy = data_sample.copy()
                    for n in range(self.num_stacks -1):
                        data_sample = np.hstack([data_sample, columns_to_copy])

                    data_pack["data_sample"] = torch.tensor(data_sample, dtype=torch.float32)
                    data_pack["class_label"] = torch.tensor(j, dtype=torch.long)
                    
                    dataset.append(data_pack)

        dataset = np.array(dataset)

        return dataset


    # Read data in from Olivia's Datasets
    def create_dataset_olivia(self):

        ActivityList = ['LocationA', 'LocationB', 'LocationC', 'LocationD', 'LocationE']
        PersonList = ['User1', 'User2', 'User3', 'User4', 'User5']
        NumSamplesPerLocation = [156, 147, 154, 160, 142, 152, 125, 155, 134, 121, 141, 100, 142, 151, 137, 145, 146, 160, 148, 144, 146, 141, 142, 141, 139]
        dataset = []
        l = -1
        for i in range(len(PersonList)):
            for j in range(len(ActivityList)):
                l +=1 
                for k in range(1, NumSamplesPerLocation[l] + 1):
                    if self.root_dir + fr"/{PersonList[i]}_{ActivityList[j]}_Normal_{k}.dat" == self.root_dir + fr"\\User4_LocationB_Normal_1.dat":
                        continue
                    data_pack = {}
                    data_sample = np.loadtxt(self.root_dir + fr"\\{PersonList[i]}_{ActivityList[j]}_Normal_{k}.dat", delimiter = ',')
                    data_pack["data_sample"] = torch.tensor(data_sample, dtype=torch.float32)
                    data_pack["class_label"] = torch.tensor(j, dtype=torch.long)
                    dataset.append(data_pack)
    
    # Read in my data from npy files to make the dataset 
    def create_dataset_maya(self):

        ActivityList = ['LocationA', 'LocationB', 'LocationC', 'LocationD', 'LocationE']
        PersonList = ['User1', 'User2', 'User3', 'User4', 'User5']
        NumSamplesPerLocation = [156, 147, 154, 160, 142, 156, 125, 155, 136, 121, 147, 101, 144, 152, 138, 149, 152, 161, 150, 146, 149, 143, 146, 155, 140]
        dataset = []
        l = -1
        for i in range(len(PersonList)):
            for j in range(len(ActivityList)):
                l +=1 
                for k in range(1, NumSamplesPerLocation[l] + 1):
                    data_pack = {}
                    data_sample = np.load(self.root_dir + fr"/{PersonList[i]}_{ActivityList[j]}_Normal_{k}.npy")
                    class_label = np.load(self.root_dir + fr"/{PersonList[i]}_{ActivityList[j]}_Label_{k}.npy")
                    data_pack["data_sample"] = torch.tensor(data_sample, dtype=torch.float32)
                    
                    data_pack["class_label"] = torch.tensor(class_label, dtype=torch.long)
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

    
 

