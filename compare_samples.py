

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


def create_dataset_normal():
    ActivityList = ['LocationA', 'LocationB', 'LocationC', 'LocationD', 'LocationE']
    PersonList = ['User1', 'User2', 'User3', 'User4', 'User5']
    dataset = []
    for i in range(len(PersonList)):
        for j in range(len(ActivityList)):
            df = df = pd.read_csv(fr'C:\\Users\\mayam\\DeepLearningFallDetection\\data' + fr"\\{PersonList[i]}_{ActivityList[j]}_Normal.csv", sep = "\t", header = 1)
            df = df[1:].astype('float64')
        
            df['Composed_Acceleration'] = np.sqrt(df['Shimmer_8665_Accel_LN_X_CAL']**2 + df['Shimmer_8665_Accel_LN_Y_CAL']**2 + df['Shimmer_8665_Accel_LN_Z_CAL']**2)
            df['Composed_Gyroscope'] = np.sqrt(np.power(df['Shimmer_8665_Gyro_X_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Y_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Z_CAL'], 2))
            normalized_acceleration = (df['Composed_Acceleration'] - df['Composed_Acceleration'].mean())/ (df['Composed_Acceleration'].std())
            normalized_gyroscope = (df['Composed_Gyroscope'] - df['Composed_Gyroscope'].mean())/ (df['Composed_Gyroscope'].std())
            peaks, _ = find_peaks(normalized_acceleration, distance = 50*2, prominence = 2)

            df['Composed_Acceleration'] = normalized_acceleration
            df['Composed_Gyroscope'] = normalized_gyroscope
            # print(len(peaks))
            k = 0
            for idx in peaks:
                data_pack_norm = {}
                data_pack_read = {}
                if (idx - ((50//2)-1)) > 0:
                    data_sample = df.iloc[idx - ((50//2-1)): idx + (50//2+1), 11:]

                    data_sample = data_sample.to_numpy()
                    columns_to_copy = data_sample.copy()
                    for i in range(1):
                        data_sample = np.hstack([data_sample, columns_to_copy])
                    
                    k +=1
                    data_pack_norm["data_sample"] = torch.tensor(data_sample, dtype=torch.float32)
                    np.savetxt(f"C:\\Users\\mayam\\DeepLearningFallDetection\\SavedData\\User1_{ActivityList[j]}_Normal_{k+1}.dat", data_pack_norm["data_sample"], delimiter=',')
                    data_pack_norm["class_label"] = torch.tensor(j, dtype=torch.long)

                    # if len(dataset) == 0:
                    #     print(data_pack_norm)
                    dataset.append(data_pack_norm)


  
    dataset = np.array(dataset)





    print("Data: ", dataset)





create_dataset_normal()

