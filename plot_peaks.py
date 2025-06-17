from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('~\\DeepLearningFallDetection\\data\\User2_LocationA_Normal.csv', sep = "\t", header = 1)

df = df[3:].astype('float64')


# df = df.drop(2, axis=0)
# print(df)

df['Composed_Acceleration'] = np.sqrt(np.power(df['Shimmer_8665_Accel_LN_X_CAL'], 2) + np.power(df['Shimmer_8665_Accel_LN_Y_CAL'], 2) + np.power(df['Shimmer_8665_Accel_LN_Y_CAL'], 2))


print(df['Composed_Acceleration'])
peaks, _ = find_peaks(df['Composed_Acceleration'], prominence = 20)


peak_values = df['Composed_Acceleration'][peaks]


plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'][3:1200],df['Composed_Acceleration'][3:1200])
plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'][peaks[:10]], peak_values[:10], "ro")
plt.show()