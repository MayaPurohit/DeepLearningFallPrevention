# Maya Purohit
# Train.py
# Used to train our models 

import os
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.abspath("CNN_Models\\AlexNet"))
sys.path.append(os.path.abspath("CNN_Models\\Initial Model"))
sys.path.append(os.path.abspath("CNN_Models\\VGGNet"))
sys.path.append(os.path.abspath("CNN_Models\\Personal_Model\\Model1"))
sys.path.append(os.path.abspath("CNN_Models\\Personal_Model\\Model2"))
sys.path.append(os.path.abspath("CNN_Models\\Personal_Model\\Model3"))
sys.path.append(os.path.abspath("LSTM_Model"))
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import wandb
import time
import psutil

from CNN_Dataloader import MotionDataset
from AlexNet import AlexNetCNN
from VGGNet import VGGNetCNN
from Test_CNN import TestCNN
from Model1 import PersModelCNN
from Model2 import SecondModelCNN
from LSTMModel import LSTM_Model
from Model3 import Model3CNN



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




    if config['model_name'] == "resnet":
        
        model = TestCNN(include_attention=config['include_attention'], input_channels= config['input_channels'],
            num_blocks=config['num_blocks'],
            num_features=config['num_features'],
        )
    elif config['model_name'] == "alex":
        model = AlexNetCNN(input_channels= config['input_channels']
        )

    elif config['model_name'] == "vgg":
        model = VGGNetCNN(input_channels= config['input_channels']
    )
    elif config['model_name'] == "model1":
        model = PersModelCNN(input_channels= config['input_channels'],
            num_features=config['num_features'],
            include_attention=config['include_attention']
        )
    elif config['model_name'] == "model2":
       model = SecondModelCNN(input_channels= config['input_channels'],
            num_features=config['num_features'], 
            num_blocks=config['num_blocks'],
            include_attention=config['include_attention']
        )
    elif config['model_name'] == "model3":
        model = Model3CNN(input_channels= config['input_channels'],
            num_features=config['num_features'],
            include_attention=config['include_attention'])
    elif config['model_name']  == "LSTM":
        model = LSTM_Model(input_size = config['input_channels'],
                          hidden_size = config['hidden_size'],
                          num_layers = config['num_layers'],
                          batch_first = config['batch_first'])




    model = model.to(device)

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
    
    # Make all necessary directories
   
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['save_dir'], config['checkpoint_dir']), exist_ok=True)
    os.makedirs(os.path.join(config['save_dir'], config['best_dir']), exist_ok=True)



    #Define datasets, dataloaders, and loss function 
    loss_fn = nn.CrossEntropyLoss()


    train_dataset = MotionDataset(config["root_dir"], config["window_size"], test_ratio = config['test_ratio'], val_ratio= config['val_ratio'], normalize=config['normalize'], input_channels=config['input_channels'],user_num= config['user_num'], mode ="train", test_type = config['test_type'])
    val_dataset = MotionDataset(config["root_dir"], config["window_size"],test_ratio = config['test_ratio'],  val_ratio= config['val_ratio'], normalize=config['normalize'], input_channels=config['input_channels'],user_num= config['user_num'], mode ="val", test_type = config['test_type'])

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=config["batch_size"],
                                              shuffle=True)
    
    # print("Training Set Size: ", len(train_dataloader.dataset))
    num_samples = len(train_dataloader.dataset)

    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=config["batch_size"],
                                              shuffle=True)
   
    train_losses = []
    best_accuracy = 0

    latency_vals = []
    throughput_vals = []
    
    
    wandb.config = {

    "epochs": config["num_epochs"],
    "batch_size": train_dataloader.batch_size,
    "learning_rate": config["learning_rate"],
    "loss_function": "Cross Entropy Loss",
    }

    process = psutil.Process()
    cpu_start = process.cpu_percent()

    for epoch in range(0, config['num_epochs']):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Learning Rate = {current_lr:.6f}")
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
        f"Train Accuracy": train_accuracy,
        "epoch": epoch + 1})
    
        scheduler.step()
        
       
        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)

        print("Training Loss: ", train_loss)

        wandb.log({"Train Loss": train_loss, "epoch": epoch + 1})
        
       #check if the model should be validated 
        should_validate = (
            config['validation_interval'] > 0 and 
            (epoch + 1) % config['validation_interval'] == 0
        ) or (epoch + 1 == config['num_epochs'])
        
        if should_validate:

            accuracy = evaluate(val_dataloader, model, num_epoch= epoch + 1)
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': accuracy,
                }, os.path.join(config['save_dir'], config['best_dir'], 'best_model.pth'))

                print(f"Saved best model with accuracy: {accuracy:.2f}")
          
        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(config['save_dir'], config['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth'))
    
    cpu_end = process.cpu_percent()
    wandb.finish()


    
    # Calculate the CPU Usage, Throughput, and  Latency 

    cpu_usage = cpu_end - cpu_start


    print("CPU Usage: ", cpu_usage)

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

    fig, ax = plt.subplots(2,1, sharex='col')
    ax[0].plot(range(len(plot_latency)), plot_latency)
    ax[1].plot(range(len(plot_through)), plot_through)

    ax[0].set_ylabel("Latency (s/sample)")
    ax[1].set_ylabel("Throughput (# samples/s)")

    fig.suptitle(f"Latency and Throughout (CPU Usage: {cpu_usage})")
    fig.supxlabel('Number of Epoch')
    plt.savefig(os.path.join(config['save_dir'], 'Latency and Throughput'))

    test()
 

def evaluate(data_loader, model, num_epoch):
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

    print("Validating")

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.eval()
    
    correct = 0.0
    total = 0.0

    
    with torch.no_grad():
        
        val_pbar = tqdm(data_loader, desc=f"Epoch {num_epoch}")
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
    
    wandb.log({"Validation Accuracy": accuracy, "epoch": num_epoch})
    
    return accuracy


def test(model = None):
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

    if model is None:
        if config['model_name'] == "resnet":
            model = TestCNN(include_attention=config['include_attention'],
                            input_channels= config['input_channels'],
                            num_blocks=config['num_blocks'],
                            num_features=config['num_features'])
        elif config['model_name'] == "alex":
            model = AlexNetCNN(input_channels= config['input_channels'])
        elif config['model_name'] == "vgg":
            model = VGGNetCNN(input_channels= config['input_channels'])
        elif config['model_name'] == "model1":
            model = PersModelCNN(input_channels= config['input_channels'],
            num_features=config['num_features'])
        elif config['model_name'] == "model2":
            model = SecondModelCNN(input_channels= config['input_channels'],
            num_features=config['num_features'],
            num_blocks=config['num_blocks'],
            include_attention=config['include_attention'])
        elif config['model_name'] == "model3":
            model = Model3CNN(input_channels= config['input_channels'],
                num_features=config['num_features'],
                include_attention=config['include_attention']
            )
        elif config['model_name']  == "LSTM":
            model = LSTM_Model(input_size = config['input_channels'],
                          hidden_size = config['hidden_size'],
                          num_layers = config['num_layers'],
                          batch_first = config['batch_first'])



        # checkpoint = torch.load(os.path.join(config['save_dir'], config['checkpoint_dir'], 'checkpoint_epoch_30.pth'))
        checkpoint = torch.load(os.path.join(config['save_dir'], config['best_dir'], f'best_model.pth'))

        model.load_state_dict(checkpoint['model_state_dict'])
        # for name, param in model.named_parameters():
        #     print(f"Layer: {name}, Shape: {param.shape}, Requires grad: {param.requires_grad}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        print(f"Loaded model from {config['best_dir']}")

    print("Testing")
    model.eval()
    
    correct = 0.0
    total = 0.0

    #Make dataset and dataloader 

    test_dataset =  MotionDataset(config["root_dir"], config["window_size"], test_ratio = config['test_ratio'],  val_ratio= config['val_ratio'], normalize=config['normalize'], input_channels=config['input_channels'], user_num= config['user_num'], mode ="test", test_type = config['test_type'])
   
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config["batch_size"],
                                              shuffle=True)
    
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

   
    precision_list = []
    recall_list = []

    # find precision and recall statistics
    for i in range(5):
        if (precision_recall[i, 2] + precision_recall[i, 0]) == 0:
            precision = 0.0
        else:
            precision = (precision_recall[i, 2]/ (precision_recall[i, 2] + precision_recall[i, 0])) * 100

        if (precision_recall[i, 2] + precision_recall[i, 1]) == 0:
            recall = 0.0
        else:
            recall = (precision_recall[i, 2]/ (precision_recall[i, 2] + precision_recall[i, 1])) * 100

        precision_list.append(precision)
        recall_list.append(recall)
        print(f'Precision on Class {i} validation images: {precision:.2f}%')
        print(f'Recall on Class {i} validation images: {recall:.2f}%')

    locations = ['Tiled Hallway', 'Carpet', 'Concrete', 'Brick', 'Lawn']

    plt.plot(locations, precision_list, label = "Precision")
    plt.plot(locations, recall_list, label = "Recall")
    plt.xlabel("Class Label")
    plt.ylabel("Percentage")
    for i, txt in enumerate(precision_list):
        plt.text(locations[i], precision_list[i], f'{txt:.2f}%')

    for i, txt in enumerate(recall_list):
        plt.text(locations[i], recall_list[i], f'{txt:.2f}%')


    
    plt.legend()
    plt.title(f"Precision and Recall Curve for Each Class, Accuracy: {accuracy:.2f}%")
    plt.savefig(os.path.join(config['save_dir'], "Precision_Recall"))

    cm = confusion_matrix(all_labels, all_outputs)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1  # prevent divide-by-zero
    cm_percent = cm.astype('float') / row_sums


   
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", xticklabels=[0, 1, 2, 3, 4], yticklabels=[0, 1, 2, 3, 4])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(config['save_dir'], "Confusion_Matrix"))
    plt.show()


    print(f'Accuracy on the {total} validation images: {accuracy:.2f}%')
    
    

    
    return accuracy



if __name__ == "__main__":
    # Configuration
    config = {

        

        'root_dir': fr'C:\\Users\\mayam\\DeepLearningFallDetection\\data',  # Training data directory
        'save_dir': fr'CNN_Models\\Personal_Model\\Model2\\Individual',  # Directory specific to the model being tested 
        
        
        # CNN Training parameters


        'batch_size': 10,                
        'num_epochs': 30,               
        'learning_rate': 1e-5,           
        'lr_decay_step': 30,            
        'lr_decay_gamma': 0.5,           
        'validation_interval': 5,        
        'input_channels'   : 2,
        'test_ratio' : 0.1,
        'val_ratio': 0.1,
        'window_size' : 50,
        'num_features': 256,             
        'num_blocks': 5,      
        'test_type' : "normal", 
        'normalize' : False,       
        'include_attention': True,
        'user_num' : 1,
        'model_name': "model2",
        

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







