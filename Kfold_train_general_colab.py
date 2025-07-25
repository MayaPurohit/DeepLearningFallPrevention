# Maya Purohit
# Train.py
# Used to train our models 

import os
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import seaborn as sns
from sklearn.metrics import confusion_matrix


base_path = "/content/drive/MyDrive/DeepLearningFallDetection"

sys.path.append(base_path)

subfolders = [
    "CNN_Models/AlexNet",
    "CNN_Models/Initial Model",
    "CNN_Models/VGGNet",
    "CNN_Models/Personal_Model/Model1",
    "CNN_Models/Personal_Model/Model2",
    "CNN_Models/Personal_Model/Model3"
    "LSTM_Model"
]

for folder in subfolders:
    full_path = os.path.join(base_path, folder)
    sys.path.append(full_path)

sys.path.append('/content/drive/MyDrive/DeepLearningFallDetection/CNN_Models/Personal_Model/Model3')


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import wandb
import time
import psutil

from CNN_Dataloader_colab import MotionDataset
from AlexNet import AlexNetCNN
from VGGNet import VGGNetCNN
from Test_CNN import TestCNN
from Model1 import PersModelCNN
from Model2 import SecondModelCNN
from Model3 import Model3CNN
from ModifiedAlex import ModAlexNetCNN
from Model2Conv2d import SecondModelCNN2d
# from LSTMModel import LSTM_Model
from sklearn.model_selection import KFold



# Metrics





def train(config):
    """
    Train the SuperResolution model
    
    Args:
        config (dict): Configuration parameters
    """
    # torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    wandb.init(project="Fall_Detection_Project", name="cnn-run")



    # Make all necessary directories
   
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['save_dir'], config['checkpoint_dir']), exist_ok=True)
    os.makedirs(os.path.join(config['save_dir'], config['best_dir']), exist_ok=True)


    # make dataset

    train_dataset = MotionDataset(config["root_dir"], config["window_size"], test_ratio = config['test_ratio'], val_ratio= config['val_ratio'], normalize=config['normalize'],num_stacks= config['num_stacks'], input_channels=config['input_channels'], user_num= config['user_num'], mode ="train", test_type = config['test_type'])

    
    #Define K-Folds Setup

    kf = KFold(n_splits=config['num_folds'], shuffle=True, random_state=42)


    fold_results = np.zeros(config['num_folds'])

            
    wandb.config = {
        "folds":  config['num_folds'],
        "epochs": config["num_epochs"],
        "batch_size": config['batch_size'],
        "learning_rate": config["learning_rate"],
        "loss_function": "Cross Entropy Loss",
    }

    fig, ax = plt.subplots(4,5, figsize=(12, 8), sharex='col', sharey='row')
    cpu_results = np.zeros(config['num_folds'])
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        print(f"\nFOLD {fold+1}/{config['num_folds']}")

    
    
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)


        train_dataloader = torch.utils.data.DataLoader(dataset=train_subset,
                                            batch_size=config["batch_size"],
                                            shuffle=True)
        num_samples = len(train_dataloader.dataset)
        val_dataloader = torch.utils.data.DataLoader(dataset=val_subset,
                                            batch_size=config["batch_size"],
                                            shuffle=True)
        
        
        if config['model_name'] == "resnet":
            
            model = TestCNN(include_attention= config['include_attention'], input_channels= config['input_channels'],
                num_blocks=config['num_blocks'],
                num_features=config['num_features'],
            )
        elif config['model_name'] == "alex":
            model = AlexNetCNN(input_channels= config['input_channels'])
        elif config['model_name'] == "modalex":
            model = ModAlexNetCNN(input_channels= config['input_channels'],
                                  num_blocks = config['num_blocks'], 
                                  include_attention = config['include_attention'])

        elif config['model_name'] == "vgg":
            model = VGGNetCNN(input_channels= config['input_channels'])
        elif config['model_name'] == "model1":
            model = PersModelCNN(input_channels= config['input_channels'],
                                 num_stacks=config['num_stacks'],
                num_features=config['num_features'],
                include_attention=config['include_attention']
            )
        elif config['model_name'] == "model2":
            model = SecondModelCNN(input_channels= config['input_channels'],
             num_stack = config['num_stacks'],
                num_features=config['num_features'], 
                num_blocks=config['num_blocks'],
                include_attention=config['include_attention']
            )
        elif config['model_name'] == "model22d":
            model = SecondModelCNN2d(input_channels= config['input_channels'],
                                   num_stack=config['num_stacks'],
                num_features=config['num_features'], 
                num_blocks=config['num_blocks'],
                include_attention=config['include_attention']
            )
        elif config['model_name']  == "LSTM":
            model = LSTM_Model(input_size = config['input_channels'],
                            hidden_size = config['hidden_size'],
                            num_layers = config['num_layers'],
                            batch_first = config['batch_first'])
        elif config['model_name'] == "model3":
            model = Model3CNN(input_channels= config['input_channels'],
                num_features=config['num_features'],
                include_attention=config['include_attention']
            )

        model = model.to(device)
        
        learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Learnable Parameters: {learnable_parameters}")
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.999)
        )

        
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_decay_step'],
            gamma=config['lr_decay_gamma']
        )

        loss_fn = nn.CrossEntropyLoss()
    
        train_losses = []
        best_accuracy = 0

        latency_vals = []
        throughput_vals = []

        process = psutil.Process()
        cpu_start = process.cpu_percent()

        for epoch in range(0, config['num_epochs']):
            model.train()
            epoch_losses = []
            
            correct = 0.0
            total = 0.0

        
            train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
            #For each batch
            for batch in train_pbar:
                
            
                data_sample = batch['data_sample'].to(device)
                
                
            
                class_label = batch['class_label'].to(device)
                
        
                optimizer.zero_grad()
                
                if config["model_name"] != "LSTM":
                    data_sample = data_sample.permute(0, 2, 1) 

                start = time.time()
                output = model(data_sample)
                end = time.time()


                latency = (end - start)/config['batch_size']
                throughput = config['batch_size']/((end - start) + (1e-5))
                latency_vals.append(latency)
                throughput_vals.append(throughput)
                output_labels = torch.argmax(output, dim = 1)

                #Not necessary to extact the class labels from the output, the loss function does it 
                loss =  loss_fn(output, class_label)
                 
                
                loss.backward()
                optimizer.step()
                
                
                epoch_losses.append(loss.item())
                train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                total += class_label.size(0)
                correct += (output_labels == class_label).sum().item()
                
            
            train_accuracy = 100 * correct / total
            print(f'Accuracy on the {total} training images: {train_accuracy:.2f}%')
    
    
            wandb.log({
            f"Train Accuracy Fold {fold+1}": train_accuracy,
            "epoch": epoch + 1})
            scheduler.step()
            
        
            train_loss = np.mean(epoch_losses)
            train_losses.append(train_loss)

            print("Training Loss: ", train_loss)

            wandb.log({f"Train Loss Fold {fold + 1}": train_loss, "epoch": epoch + 1})
            
            should_validate = (
                config['validation_interval'] > 0 and 
                (epoch + 1) % config['validation_interval'] == 0
            ) or (epoch + 1 == config['num_epochs'])
            
            if should_validate:
                accuracy = evaluate(val_dataloader, model, fold_num=fold, num_epoch=epoch + 1)
                if accuracy >= best_accuracy:
                        fold_results[fold] = accuracy
                        best_accuracy = accuracy
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'accuracy': accuracy,
                        }, os.path.join(config['save_dir'], config['best_dir'], f'best_model{fold}.pth'))

                        print(f"Saved best model with accuracy: {accuracy:.2f}")
        cpu_end = process.cpu_percent()

        cpu_usage = cpu_end - cpu_start


        print("CPU Usage: ", cpu_usage)
        cpu_results[fold] = cpu_usage

        sum_lat = 0
        sum_thr = 0
        plot_latency = []
        plot_through = []
        for i in range(len(latency_vals)):
            sum_lat +=latency_vals[i]
            sum_thr += throughput_vals[i]
            if i % int(num_samples/config['batch_size']) == 0:
                lat_avg = sum_lat/(num_samples/config['batch_size'])
                thr_avg = sum_thr/(num_samples/config['batch_size'])
                plot_latency.append(lat_avg)
                plot_through.append(thr_avg)
                sum_lat = 0
                sum_thr = 0

    
        #plot the values 

        
        plt.figure(fig.number)
        if fold >= int(config['num_folds'] / 2):
            ax[2, fold -  int(config['num_folds'] / 2)].plot(range(len(plot_latency)), plot_latency, label = "Latency")
            ax[3, fold -  int(config['num_folds'] / 2)].plot(range(len(plot_through)), plot_through, label = "Throughout", color = "r")
            ax[2, fold -  int(config['num_folds'] / 2)].set_title(f'Model {fold} CPU: {cpu_usage}%')
            ax[2, fold -  int(config['num_folds'] / 2)].legend()
            ax[3, fold -  int(config['num_folds'] / 2)].legend()
            if (fold -  int(config['num_folds'] / 2) == 0):
                ax[2, fold -  int(config['num_folds'] / 2)].set_ylabel("s/sample")
                ax[3, fold -  int(config['num_folds'] / 2)].set_ylabel("# samples/s")
        else:
            ax[0, fold].plot(range(len(plot_latency)), plot_latency, label = "Latency")
            ax[1, fold].plot(range(len(plot_through)), plot_through, label = "Throughput", color = "r")
            ax[0, fold].set_title(f'Model {fold} CPU: {cpu_usage}%')
            ax[0, fold].legend()
            ax[1, fold].legend()
            if fold == 0:
                ax[0, fold].set_ylabel("Latency (s/sample)")
                ax[1, fold].set_ylabel("Throughput (# samples/s)")

    average_cpu = np.mean(cpu_results)
    fig.suptitle(f"Latency and Throughout: Parameters {learnable_parameters}")
    fig.supxlabel('Number of Epoch')
    plt.savefig(os.path.join(config['save_dir'], 'Latency and Throughput Overall'))
    plt.close(fig)

    plt.figure()
    plt.plot(range(1, fold_results.shape[0]+1), fold_results)
    plt.xticks(ticks=range(1, fold_results.shape[0]+1))
    plt.xlabel("Fold Number")
    plt.ylabel("Percentage Accuracy")
    plt.title("Cross-Validation Fold Accuracy for Overall Model")
    if config['test_type'] == "individual":
        plt.savefig(os.path.join(config['save_dir'], f"Fold Accuracy User {config['user_num']}"))
    else:
        plt.savefig(os.path.join(config['save_dir'], "Fold Accuracy Overall"))
    plt.clf() 

    overall_accuracy = test()

    print(f"Overall Accuracy: {overall_accuracy}")
    print(f"Overall CPU Usage: {average_cpu}")


            

    

def evaluate(data_loader, model, fold_num, num_epoch):
    """
    Evaluate the Siamese network
    
    Args:
        args: Command line arguments
        split: Data split ('training' or 'testing')
        data_loader: DataLoader for the split
        siamese_net: Trained Siamese network
        visualize: Whether to visualize predictions
    """
    # Set model to evaluation mode

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print("Validating")
    model.eval()
    
    correct = 0.0
    total = 0.0

    
    with torch.no_grad():
        
        val_pbar = tqdm(data_loader, desc=f"Fold {fold_num}")
        for data_samples in val_pbar:
            data = data_samples["data_sample"].to(device)
            labels = data_samples["class_label"].to(device)
            
            if config["model_name"] != "LSTM":
                data = data.permute(0, 2, 1) 
            
            #Check to see how many samples were classified correctly 
            output_labels = model(data)
            _, outputs = torch.max(output_labels, dim = 1)

            total += labels.size(0)
            correct += (outputs == labels).sum().item()
    
    #Calculate accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy on the {total} validation images: {accuracy:.2f}%')
    
    model.train()
    
    wandb.log({f"Validation Accuracy Fold {fold_num + 1}": accuracy, "epoch": num_epoch})
    
    return accuracy


def test(user_ind = None, model = None):
    """
    Evaluate the Siamese network
    
    Args:
        args: Command line arguments
        split: Data split ('training' or 'testing')
        data_loader: DataLoader for the split
        siamese_net: Trained Siamese network
        visualize: Whether to visualize predictions
    """
    # Set Model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    if model is None:
        if config['model_name'] == "resnet":
            model = TestCNN(include_attention= config['include_attention'],input_channels= config['input_channels'],
                num_blocks=config['num_blocks'],
                num_features=config['num_features'])
        elif config['model_name'] == "alex":
            model = AlexNetCNN(input_channels= config['input_channels'])
        elif config['model_name'] == "modalex":
            model = ModAlexNetCNN(input_channels= config['input_channels'],
                                  num_blocks = config['num_blocks'], 
                                  include_attention = config['include_attention'])
        elif config['model_name'] == "vgg":
            model = VGGNetCNN(input_channels= config['input_channels'])
        elif config['model_name'] == "model1":
            model = PersModelCNN(input_channels= config['input_channels'],
                                  num_stacks=config['num_stacks'],
            num_features=config['num_features'])
        elif config['model_name'] == "model2":
            model = SecondModelCNN(input_channels= config['input_channels'],
                                    num_stack = config['num_stacks'],
            num_features=config['num_features'],
            num_blocks=config['num_blocks'],
            include_attention=config['include_attention'])
        elif config['model_name'] == "model22d":
            model = SecondModelCNN2d(input_channels= config['input_channels'],
                                   num_stack=config['num_stacks'],
                num_features=config['num_features'], 
                num_blocks=config['num_blocks'],
                include_attention=config['include_attention']
            )
        elif config['model_name']  == "LSTM":
            model = LSTM_Model(input_size = config['input_channels'],
                          hidden_size = config['hidden_size'],
                          num_layers = config['num_layers'],
                          batch_first = config['batch_first'])
        elif config['model_name'] == "model3":
            model = Model3CNN(input_channels= config['input_channels'],
                num_features=config['num_features'],
                include_attention=config['include_attention']
            )

    model = model.to(device)
    #Make dataset and dataloader 

    test_dataset =  MotionDataset(config["root_dir"], config["window_size"], test_ratio = config['test_ratio'],  val_ratio= config['val_ratio'], normalize=config['normalize'], input_channels=config['input_channels'], user_num= config['user_num'], num_stacks= config['num_stacks'], mode ="test", test_type = config['test_type'])

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=config["batch_size"],
                                            shuffle=True)
    

    fig_prec, axes_prec = plt.subplots(2, (int(config['num_folds'] / 2)), figsize=(15, 6), sharex = 'col')  
    fig_conf, axes_conf = plt.subplots(2, (int(config['num_folds'] / 2)), figsize=(15, 6), sharex = 'col')

    accuracy_results = np.zeros(config['num_folds'])
    for j in range(config['num_folds']):
        checkpoint = torch.load(os.path.join(config['save_dir'], config['best_dir'], f'best_model{j}.pth'))


        model.load_state_dict(checkpoint['model_state_dict'])
        learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Loaded model from {config['best_dir']}, Parameter: {learnable_params}")

        print("Testing")
        model.eval()
        
        correct = 0.0
        total = 0.0

    
        
        precision_recall = np.zeros([5, 3]) # 5 for the number of classes, 4 for the true/false positive/negative
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            test_pbar = tqdm(test_dataloader, desc=f"Test")
            for data_samples in test_pbar:
                data_samples["data_sample"] = data_samples["data_sample"].to(device)
                data_samples["class_label"] = data_samples["class_label"] .to(device)
                
                
                data = data_samples["data_sample"]
                labels = data_samples["class_label"]
                if config["model_name"] != "LSTM":
                    data = data.permute(0, 2, 1) 
                
                
                output_labels = model(data)
                _, outputs = torch.max(output_labels, dim = 1)

                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                #Precision and Recall calculations

                for i in range(5):
                    precision_recall[i,0] += (labels[outputs == i] != i).sum().item() #false positive
                    precision_recall[i,1] += (labels[outputs != i] == i).sum().item() #false negative 
                    precision_recall[i,2] += (labels[outputs == i] == i).sum().item() #true positive


                total += labels.size(0)
                correct += (outputs == labels).sum().item()
        
        # calculate accuracy
        accuracy = 100 * correct / total
        accuracy_results[j] = accuracy

    
        precision_list = []
        recall_list = []

        # find precision and recall statistics
        for i in range(5):
            if (precision_recall[i, 2] + precision_recall[i, 0]) == 0:
                precision = 0.0
            else:
                precision = (precision_recall[i, 2]/ (precision_recall[i, 2] + precision_recall[i, 0])) * 100
            recall = (precision_recall[i, 2]/ (precision_recall[i, 2] + precision_recall[i, 1])) * 100
            precision_list.append(precision)
            recall_list.append(recall)
            # print(f'Precision on Class {i} validation images: {precision:.2f}%')
            # print(f'Recall on Class {i} validation images: {recall:.2f}%')

        locations = ['TH', 'Ca', 'Con', 'B', 'L']

        if j >= int(config['num_folds'] / 2):
            axes_prec[1, j - (int(config['num_folds'] / 2))].plot(locations, precision_list, label = "Precision")
            axes_prec[1, j - (int(config['num_folds'] / 2))].plot(locations, recall_list, label = "Recall")
            # for i, txt in enumerate(precision_list):
            #     axes_prec[1, j - (int(config['num_folds'] / 2))].text(locations[i], precision_list[i], f'{txt:.2f}%')

            # for i, txt in enumerate(recall_list):
            #     axes_prec[1, j - (int(config['num_folds'] / 2))].text(locations[i], recall_list[i], f'{txt:.2f}%')
            axes_prec[1, j - (int(config['num_folds'] / 2))].set_title(f'Model {j} {accuracy:.2f}%')
            axes_prec[1, j - (int(config['num_folds'] / 2))].legend()
        else:
            axes_prec[0, j].plot(locations, precision_list, label = "Precision")
            axes_prec[0, j].plot(locations, recall_list, label = "Recall")
            # for i, txt in enumerate(precision_list):
            #     axes_prec[0, j].text(locations[i], precision_list[i], f'{txt:.2f}%')

            # for i, txt in enumerate(recall_list):
            #     axes_prec[0, j].text(locations[i], recall_list[i], f'{txt:.2f}%')
            axes_prec[0,j].set_title(f'Model {j} {accuracy:.2f}%')
            axes_prec[0, j].legend()
      
        #Confusion Matrix 
        cm = confusion_matrix(all_labels, all_outputs)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    
        if j >= int(config['num_folds'] / 2):
            sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", xticklabels=[0, 1, 2, 3, 4], yticklabels=[0, 1, 2, 3, 4], ax = axes_conf[1, j - (int(config['num_folds'] / 2))])
            axes_conf[1, j - (int(config['num_folds'] / 2))].set_title(f"Model {j}")  
        else:
            sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", xticklabels=[0, 1, 2, 3, 4], yticklabels=[0, 1, 2, 3, 4], ax = axes_conf[0, j])
            axes_conf[0, j].set_title(f"Model {j}")  
    
    plt.figure(fig_prec.number)
    fig_prec.supxlabel("Class Label")
    fig_prec.supylabel("Percentage")

    fig_prec.suptitle(f"Precision and Recall Curve for Each Class: Attention: {config['include_attention']}")
    if config['test_type'] == "individual":
        fig_prec.savefig(os.path.join(config['save_dir'], f"Precision_Recall (K-Fold) User {config['user_num']}"))
    else:
        fig_prec.savefig(os.path.join(config['save_dir'], "Precision_Recall (K-Fold) Overall"))
   
    plt.close(fig_prec)

    fig_conf.supxlabel("Predicted Label")
    fig_conf.supylabel("True Label")

    fig_conf.suptitle("Confusion Matrix")
    plt.figure(fig_conf.number)
    if config['test_type'] == "individual":
        fig_conf.savefig(os.path.join(config['save_dir'], f"Confusion_Matrix (K-Fold) User {config['user_num']}"))
    else:
        fig_conf.savefig(os.path.join(config['save_dir'], "Confusion_Matrix (K-Fold) Overall"))

    plt.close(fig_conf)

    

    
    return np.mean(accuracy_results)



if __name__ == "__main__":
    # Configuration
    config = {

        

        'root_dir': "/content/drive/MyDrive/DeepLearningFallDetection/data",
        'save_dir': "/content/drive/MyDrive/DeepLearningFallDetection/CNN_Models/Personal_Model/Model2/TestGeneral",  # Directory specific to the model being tested
        
        
        # CNN Training parameters


        'batch_size': 10,                
        'num_epochs': 50,               
        'learning_rate': 1e-3,           
        'lr_decay_step': 10,            
        'lr_decay_gamma': 0.5,           
        'validation_interval': 5,        
        'input_channels'   : 2,
        'test_ratio' : 0.1,
        'val_ratio': 0,
        'window_size' : 50,
        'num_features': 50,             
        'num_blocks': 2,      
        'test_type' : "normal", 
        'normalize' : False,       
        'include_attention': False,
        'model_name': "model22d",
        'num_folds' : 10,
        'user_num' : [1,2,3,4,5],
        'num_stacks': 2,
        

        # LSTM Training parameters
        'hidden_size': 64,
        'num_layers': 32,
        'batch_first': True,

        'checkpoint_dir': 'checkpoints', 
        'run_type' : 'train',
        'best_dir': 'best_model',         # Directory to save sample images
        'save_every': 5,                 # Save checkpoint every N epochs
        'resume': None,                  # Path to checkpoint to resume from
    }


    
    

    if config['run_type'] == 'test':
        test()
    else:
        train(config)







