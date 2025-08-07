from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

# df = pd.read_csv('~\\DeepLearningFallDetection\\data\\User1_LocationB_Normal.csv', sep = "\t", header = 1)
# # np.set_printoptions(threshold=np.inf)

# df = df.iloc[1:].astype('float64')


# df['Composed_Acceleration'] = np.sqrt(df['Shimmer_8665_Accel_LN_X_CAL']**2 + df['Shimmer_8665_Accel_LN_Y_CAL']**2 + df['Shimmer_8665_Accel_LN_Z_CAL']**2)

# df['Composed_Gyroscope'] = np.sqrt(np.power(df['Shimmer_8665_Gyro_X_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Y_CAL'], 2) + np.power(df['Shimmer_8665_Gyro_Z_CAL'], 2))



# normalized_acceleration = (df['Composed_Acceleration'] - df['Composed_Acceleration'].mean())/ (df['Composed_Acceleration'].std())
# print(normalized_acceleration)
# peaks, _ = find_peaks(normalized_acceleration, distance = 100, prominence = 2)




# weights = np.zeros_like(df['Composed_Acceleration'], dtype=float)

# window = 50 
# decay = 0.005

# # Apply weights around each peak
# for peak in peaks:
#     for i in range(-window, window + 1):
#         idx = peak + i
#         if 0 <= idx < len(df['Composed_Acceleration']):
#             weights[idx] += np.exp(-decay * abs(i))

# # Enhance the signal
# enhanced_signal = df['Composed_Acceleration'].to_numpy() * weights



# peak_values = df['Composed_Acceleration'].iloc[peaks]


# plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'], enhanced_signal)

# plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'], df['Composed_Acceleration'])
# # plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'].iloc[peaks], peak_values, "ro")
# # plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'].iloc[peaks[0] - 67], df['Composed_Acceleration'].iloc[peaks[0] - 67], "bo")
# # plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'].iloc[peaks[0] + 67], df['Composed_Acceleration'].iloc[peaks[0] + 68], "bo")
# # plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'].iloc[peaks[1] - 67], df['Composed_Acceleration'].iloc[peaks[1] - 67], "go")
# # plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'].iloc[peaks[1] + 68], df['Composed_Acceleration'].iloc[peaks[1] + 68], "go")
# # plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'].iloc[peaks[2] - 67], df['Composed_Acceleration'].iloc[peaks[2] - 67], "ko")
# # plt.plot(df['Shimmer_8665_Timestamp_Unix_CAL'].iloc[peaks[2] + 68], df['Composed_Acceleration'].iloc[peaks[2] + 68], "ko")
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


# acc_data = np.array([[1,1,1,1,1],
#                 [0.60533333,	0.67391304,	0.61617647,	0.716,	0.77808219],
#                 [1,	1,	0.9765,	0.9773,	1],
#                 [1,1,1,1,1]])

# cpu_data =  np.array([[553.94,	553.94,	553.94,	553.94,	553.94],
#                 [196.38, 194.63, 183.75, 181.8,  213.61],
#                 [236.82,	274.51,	313.1,	279.77,	299.76],
#                 [507.65,	500.33,	514.41,	513.27,	517.15]])
# row_labels = ['Alex', 'Light', 'Middle', 'Heavy']
# col_labels = ['User1', 'User2', 'User3', 'User4', 'User5']

# # 2. Create Heatmap

# plt.imshow(cpu_data, cmap='inferno')

# # 3. Set Labels
# plt.xticks(np.arange(len(col_labels)), col_labels)
# plt.yticks(np.arange(len(row_labels)), row_labels)
# plt.colorbar()
# plt.title("CPU Usage Percentage for Individual Users")
# plt.savefig("Heatmap CPU")
# plt.show()



# Sample data
labels = ['Personalized Model', 'Generalized Model']
group1 = [0.9152,0.8686]
group2 = [0.8326, 0.8019]

colors = ['#66c2a5', '#fc8d62'] 

x = np.arange(len(labels))  # label locations
width = 0.35  # width of each bar

# Create the plot
fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, group1, width, label='Our Model', color = colors[0])
bars2 = ax.bar(x + width/2, group2, width, label='AlexNet', color = colors[1])


for i, bar1 in enumerate(bars1):
    height = bar1.get_height()
    plt.text(bar1.get_x() + bar1.get_width()/2, height + 0.005,  # +1 to move text above bar
            f'{group1[i]}', ha='center', va='bottom')
    

for j, bar2 in enumerate(bars2):
    height = bar2.get_height()
    plt.text(bar2.get_x() + bar2.get_width()/2, height + 0.005,  # +1 to move text above bar
            f'{group2[j]}', ha='center', va='bottom')
   

# Add labels, title, legend
ax.set_xlabel('Type of Model')
ax.set_ylabel('Accuracy')
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.tight_layout()
plt.show()

plt.figure()


categories = ['MLP', 'SMO','5NN','J48', 'Our Model']
values = [0.815, 0.794, 0.75, 0.671, 0.915]

# Colors — you can pick any, here using Set2 from seaborn
colors = sns.color_palette("Set2", n_colors=len(categories))

# Create bar plot
bars = plt.bar(categories, values, color=colors)

# Add labels on top of each bar
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.02,
             f'{values[i]}', ha='center', va='bottom')

# Labels and title
plt.xlabel('Type of Model')
plt.ylabel('Accuracy')
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# plt.legend()

plt.show()




# plt.xlabel('Generalized Model Architecture')

# plt.ylabel('Accuracy')


# plt.show()
