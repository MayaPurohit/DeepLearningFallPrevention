from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from findpeaks import findpeaks

# df = pd.read_csv('~\\DeepLearningFallDetection\\data\\User1_LocationB_Normal.csv', sep = "\t", header = 1)
# # np.set_printoptions(threshold=np.inf)

# df = df.iloc[1:].astype('float64')


# df['Composed_Acceleration'] = np.sqrt(df['Shimmer_8665_Accel_LN_X_CAL']**2 + df['Shimmer_8665_Accel_LN_Y_CAL']**2 + df['Shimmer_8665_Accel_LN_Z_CAL']**2)

# df['Composed_Gyroscope'] = np.sqrt(np.power(df['Shimmer_8665_Gyro_X_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Y_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Z_CAL'], 2))



# normalized_acceleration = (df['Composed_Acceleration'] - df['Composed_Acceleration'].mean())/ (df['Composed_Acceleration'].std())
# print(normalized_acceleration)
# peaks, _ = find_peaks(normalized_acceleration, distance = 100, prominence = 2)
# # second_derivative = np.gradient(np.gradient(normalized_acceleration, df['Shimmer_8665_Timestamp_Unix_CAL']), df['Shimmer_8665_Timestamp_Unix_CAL'])


# peak_values = df['Composed_Acceleration'].iloc[peaks]


# threshold = 0.01 * np.max(np.abs(second_derivative))  

# second_derivative[np.abs(second_derivative) < threshold] = 0


# sign_change = np.where(np.diff(np.sign(second_derivative)) != 0)[0]

# print(np.diff(np.sign(second_derivative)))
# inds = np.where(np.diff(sign_change) > 5)
# print(np.diff(sign_change))





# peak_values = df['Composed_Acceleration'].iloc[peaks]


# plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'],df['Composed_Acceleration'])
# plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'].iloc[peaks], peak_values, "ro")
# plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'].iloc[peaks[0] - 67], df['Composed_Acceleration'].iloc[peaks[0] - 67], "bo")
# plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'].iloc[peaks[0] + 67], df['Composed_Acceleration'].iloc[peaks[0] + 68], "bo")
# plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'].iloc[peaks[1] - 67], df['Composed_Acceleration'].iloc[peaks[1] - 67], "go")
# plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'].iloc[peaks[1] + 68], df['Composed_Acceleration'].iloc[peaks[1] + 68], "go")
# plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'].iloc[peaks[2] - 67], df['Composed_Acceleration'].iloc[peaks[2] - 67], "ko")
# plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'].iloc[peaks[2] + 68], df['Composed_Acceleration'].iloc[peaks[2] + 68], "ko")
# plt.show()




# models = ['Model 1', 'Model 2', 'Model 3', 'AlexNet', 'ResNet (0)', 'ResNet (1)']
# parameters = [989317, 3636517, 594949,19680261, 51077, 199685]
# colors_bar = ['red', 'green', 'blue', 'purple', 'orange', 'black']
# plt.bar(models, parameters, color = colors_bar)
# plt.title('Model Parameter Comparison')
# plt.xlabel('Model Types')
# plt.ylabel('# of Parameters')
# plt.savefig("Parameter Comparison")


# plt.figure()


# models = ['Model 1', 'Model 2', 'Model 3', 'AlexNet', 'ResNet (0)', 'ResNet (1)']
# parameters = [989317, 3636517, 594949,19680261, 51077, 199685]
# colors_bar = ['red', 'green', 'blue', 'purple', 'orange', 'black']
# plt.bar(models, parameters, color = colors_bar)
# plt.title('Model Parameter Comparison')
# plt.xlabel('Model Types')
# plt.ylabel('Time taken (sec)')
# plt.savefig("Parameter Comparison")


acc_data = np.array([[1,1,1,1,1],
                [0.60533333,	0.67391304,	0.61617647,	0.716,	0.77808219],
                [1,	1,	0.9765,	0.9773,	1],
                [1,1,1,1,1]])

cpu_data =  np.array([[553.94,	553.94,	553.94,	553.94,	553.94],
                [196.38, 194.63, 183.75, 181.8,  213.61],
                [236.82,	274.51,	313.1,	279.77,	299.76],
                [507.65,	500.33,	514.41,	513.27,	517.15]])
row_labels = ['Alex', 'Light', 'Middle', 'Heavy']
col_labels = ['User1', 'User2', 'User3', 'User4', 'User5']

# 2. Create Heatmap

plt.imshow(cpu_data, cmap='inferno')

# 3. Set Labels
plt.xticks(np.arange(len(col_labels)), col_labels)
plt.yticks(np.arange(len(row_labels)), row_labels)
plt.colorbar()
plt.title("CPU Usage Percentage for Individual Users")
plt.savefig("Heatmap CPU")
plt.show()