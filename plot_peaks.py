from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('~\\DeepLearningFallDetection\\data\\User2_LocationA_Normal.csv', sep = "\t", header = 1)


df = df.iloc[1:].astype('float64')


df['Composed_Acceleration'] = np.sqrt(df['Shimmer_8665_Accel_LN_X_CAL']**2 + df['Shimmer_8665_Accel_LN_Y_CAL']**2 + df['Shimmer_8665_Accel_LN_Z_CAL']**2)


peaks, _ = find_peaks(df['Composed_Acceleration'], threshold = 3, distance = 128, prominence = 12 )


peak_values = df['Composed_Acceleration'].iloc[peaks]


plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'],df['Composed_Acceleration'])
plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'].iloc[peaks], peak_values, "ro")
plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'].iloc[peaks[0] - 67], df['Composed_Acceleration'].iloc[peaks[0] - 67], "bo")
plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'].iloc[peaks[0] + 67], df['Composed_Acceleration'].iloc[peaks[0] + 68], "bo")
plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'].iloc[peaks[1] - 67], df['Composed_Acceleration'].iloc[peaks[1] - 67], "go")
plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'].iloc[peaks[1] + 68], df['Composed_Acceleration'].iloc[peaks[1] + 68], "go")

# for idx in peaks:
#         data_sample = df.iloc[idx - ((128//2)-1): idx + (128//2), 0:7]
#         print(data_sample)
#         plt.plot(data_sample['Shimmer_8665_Timestamp_Unix_CAL'].iloc[1:idx - ((128//2)-1)], data_sample['Shimmer_8665_Accel_LN_X_CAL'].iloc[1:idx - ((128//2)-1)])
plt.show()