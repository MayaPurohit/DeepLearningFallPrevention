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

sys.path.append(os.path.abspath("CNN_Models\\AlexNet"))
sys.path.append(os.path.abspath("CNN_Models\\Initial Model"))
sys.path.append(os.path.abspath("CNN_Models\\VGGNet"))
sys.path.append(os.path.abspath("CNN_Models\\Personal_Model"))
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import wandb
import time
import psutil

from dataloader import MotionDataset
from AlexNet import AlexNetCNN
from VGGNet import VGGNetCNN
from Test_CNN import TestCNN
from PersModel import PersModelCNN


# Metrics





def train(config, test_type, model_name):
    """
    Train the SuperResolution model
    
    Args:
        config (dict): Configuration parameters
    """
    # torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    wandb.init(project="Fall_Detection_Project", name="cnn-run")




    if model_name == "initial":
        
        model = TestCNN(input_channels= config['input_channels'],
            num_blocks=config['num_blocks'],
            num_features=config['num_features'],
        )
    elif model_name == "alex":
        model = AlexNetCNN(input_channels= config['input_channels']
        )

    elif model_name == "vgg":
        model = VGGNetCNN(input_channels= config['input_channels']
    )
    elif model_name == "personal":
        model = PersModelCNN(input_channels= config['input_channels'],
            num_features=config['num_features']
        )



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


    train_dataset = MotionDataset(config["root_dir"], config["window_size"], test_ratio = config['test_ratio'], mode ="train", test_type = test_type)
    val_dataset = MotionDataset(config["root_dir"], config["window_size"],test_ratio = config['test_ratio'], mode ="val", test_type = test_type)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=config["batch_size"],
                                              shuffle=True)
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
        model.train()
        epoch_losses = []
        
       
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        #For each batch
        for batch in train_pbar:
            
          
            data_sample = batch['data_sample'].to(device)
            class_label = batch['class_label'].to(device)
            
       
            optimizer.zero_grad()
            
            
            data_sample = data_sample.permute(0, 2, 1) 
            start = time.time()
            output = model(data_sample)
            
            end = time.time()
           
            latency = (end - start)/config['batch_size']
            throughput = config['batch_size']/((end - start) + (1e-5))
            latency_vals.append(latency)
            throughput_vals.append(throughput)
    

            loss =  loss_fn(output, class_label)
            
            
            loss.backward()
            optimizer.step()
            
            
            epoch_losses.append(loss.item())
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        
        scheduler.step()
        
       
        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)

        print("Training Loss: ", train_loss)

        wandb.log({"train_loss": train_loss, "epoch": epoch + 1})
        
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
    model.eval()
    
    correct = 0.0
    total = 0.0

    
    with torch.no_grad():
        
        val_pbar = tqdm(data_loader, desc=f"Epoch {num_epoch}")
        for data_samples in val_pbar:
            
            if torch.cuda.is_available():
                data_samples["data_sample"] = data_samples["data_sample"].cuda()
                data_samples["class_label"] = data_samples["class_label"] .cuda()
            
            
            data = data_samples["data_sample"]
            labels = data_samples["class_label"]
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


def test(test_type, model_name, model = None):
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
        if model_name == "initial":
            model = TestCNN(input_channels= config['input_channels'],
                num_blocks=config['num_blocks'],
                num_features=config['num_features'])
        elif model_name == "alex":
            model = AlexNetCNN(input_channels= config['input_channels'])
        elif model_name == "vgg":
            model = VGGNetCNN(input_channels= config['input_channels'])
        elif model_name == "personal":
            model = PersModelCNN(input_channels= config['input_channels'],
            num_features=config['num_features'])


        checkpoint = torch.load(os.path.join(config['save_dir'], config['best_dir'], 'best_model.pth'))


        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {config['best_dir']}")

    print("Testing")
    model.eval()
    
    correct = 0.0
    total = 0.0

    #Make dataset and dataloader 

    test_dataset =  MotionDataset(config["root_dir"], config["window_size"], test_ratio = config['test_ratio'], mode ="test", test_type = test_type)
   
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config["batch_size"],
                                              shuffle=True)
    
    precision_recall = np.zeros([5, 4]) # 5 for the number of classes, 4 for the true/false positive/negative
    with torch.no_grad():
        test_pbar = tqdm(test_dataloader, desc=f"Test")
        for data_samples in test_pbar:
            
            if torch.cuda.is_available():
                data_samples["data_sample"] = data_samples["data_sample"].cuda()
                data_samples["class_label"] = data_samples["class_label"] .cuda()
            
            
            data = data_samples["data_sample"]
            labels = data_samples["class_label"]
            data = data.permute(0, 2, 1) 
            
            output_labels = model(data)
            _, outputs = torch.max(output_labels, dim = 1)


            #Precision and Recall calculations

            for i in range(5):
                precision_recall[i,0] += (labels[outputs == i] != i).sum().item() #false positive
                precision_recall[i,1] += (labels[outputs != i] == i).sum().item() #false negative 
                precision_recall[i,2] += (labels[outputs != i] != i).sum().item() #true negative
                precision_recall[i,3] += (labels[outputs == i] == i).sum().item() #true positive


            total += labels.size(0)
            correct += (outputs == labels).sum().item()
    
    # calculate accuracy
    accuracy = 100 * correct / total

   
    precision_list = []
    recall_list = []
    accuracy_list = []
    # find precision and recall statistics
    for i in range(5):
        if (precision_recall[i, 3] + precision_recall[i, 0]) == 0:
            precision = 0.0
        else:
            precision = precision = (precision_recall[i, 3]/ (precision_recall[i, 3] + precision_recall[i, 0])) * 100
        recall = (precision_recall[i, 3]/ (precision_recall[i, 3] + precision_recall[i, 1])) * 100
        accuracy = ((precision_recall[i, 3] +precision_recall[i,2]) / (precision_recall[i, 0]+ precision_recall[i, 2] + precision_recall[i, 3] + precision_recall[i, 1])) * 100
        precision_list.append(precision)
        recall_list.append(recall)
        accuracy_list.append(accuracy)
        print(f'Precision on Class {i} validation images: {precision:.2f}%')
        print(f'Recall on Class {i} validation images: {recall:.2f}%')
        print(f'Accuracy on Class {i} validation images: {accuracy:.2f}%')

    locations = ['Tiled Hallway', 'Carpet', 'Concrete', 'Brick', 'Lawn']

    plt.plot(locations, precision_list, label = "Precision")
    plt.plot(locations, recall_list, label = "Recall")
    plt.plot(locations, accuracy_list, label = "Accuracy")
    plt.xlabel("Class Label")
    plt.ylabel("Percentage")
    for i, txt in enumerate(precision_list):
        plt.text(locations[i], precision_list[i], f'{txt:.2f}%')

    for i, txt in enumerate(recall_list):
        plt.text(locations[i], recall_list[i], f'{txt:.2f}%')

    for i, txt in enumerate(accuracy_list):
        plt.text(locations[i], accuracy_list[i], f'{txt:.2f}%')
    
    plt.legend()
    plt.title("Accuracy, Precision and Recall Curve for Each Class")
    plt.savefig(os.path.join(config['save_dir'], "Accuracy_Precision_Recall"))


    print(f'Accuracy on the {total} validation images: {accuracy:.2f}%')
    
    

    
    return accuracy



if __name__ == "__main__":
    # Configuration
    config = {

        

        'root_dir': fr'~\\DeepLearningFallDetection\\data',  # Training data directory
        'save_dir': fr'CNN_Models\\Personal_Model',  # Directory specific to the model being tested 
        
        
        # Training parameters
        'batch_size': 15,                
        'num_epochs': 150,               
        'learning_rate': 5e-6,           
        'lr_decay_step': 30,            
        'lr_decay_gamma': 0.5,           
        'validation_interval': 5,        
        'input_channels'   : 6,
        'test_ratio' : 0.2,
        'window_size' : 128,
        'num_features': 64,             
        'num_blocks': 8,      
        'test_type' : "user",         
        

        'checkpoint_dir': 'checkpoints', 
        'run_type' : 'test',
        'best_dir': 'best_model',         # Directory to save sample images
        'save_every': 5,                 # Save checkpoint every N epochs
        'resume': None,                  # Path to checkpoint to resume from
    }
    

    if config['run_type'] == 'test':
        test(config['test_type'], "personal")
    else:
        train(config, config['test_type'], "personal")