## Surface Classification for five different surfaces using CNN Architecture

# Dataset

Five different users walk on the following surfaces: 
- Tiled Hallway
- Carpeted Floor
- Concrete Pavement
- Brick Road
- Lawn



# Directory Layout:

-DeepLearningFallPrevention-main
    |   README.md
    |
    +---CNN_Models
    |   +---AlexNet
    |   |   |   AlexNet.py
    |   |   |   ModifiedAlex.py
    |   |   |
    |   |   \---__pycache__
    |   |           AlexNet.cpython-312.pyc
    |   |
    |   +---Initial Model
    |   |   |   Test_CNN.py
    |   |   |
    |   |   \---__pycache__
    |   |           Existing_CNN.cpython-312.pyc
    |   |           Test_CNN.cpython-312.pyc
    |   |
    |   +---Personal_Model
    |   |   +---Model1
    |   |   |   |   Model1.py
    |   |   |   |
    |   |   |   \---__pycache__
    |   |   |           PersModel.cpython-312.pyc
    |   |   |
    |   |   +---Model2
    |   |   |   |   Model2.py
    |   |   |   |   Model2Conv2d.py
    |   |   |   |
    |   |   |   \---__pycache__
    |   |   |           SecondModel.cpython-312.pyc
    |   |   |
    |   |   \---Model3
    |   |           Model3.py
    |   |
    |   \---VGGNet
    |           VGGNet.py
    |
    +---Dataloaders
    |       CNN_Dataloader.py
    |       CNN_Dataloader_colab.py
    |       CNN_Dataloader_colab_transfer.py
    |
    +---Debug
    |       compare_samples.py
    |       plot_peaks.py
    |
    +---LSTM_Model
    |       LSTMModel.py
    |
    \---Training Scripts
            Kfold_train.py
            Kfold_train_colab.py
            Kfold_train_colab_general_new.py
            Kfold_train_colab_new.py
            Kfold_train_general.py
            Kfold_train_general_colab.py
            train.py
            train_colab.py
            transfer_learning_training.py
# To run:

  
