# Maya Purohit
#4/19/2025
# Train.py
# Train the SRCNN Model

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import wandb

from dataloader import MotionDataset
from Existing_CNN import TestCNN


# Metrics


#10%
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

    model = TestCNN(input_channels= config['input_channels'],
        num_blocks=config['num_blocks'],
        num_features=config['num_features'],
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
    
   
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['best_dir'], exist_ok=True)
    

    loss_fn = nn.CrossEntropyLoss()
    train_dataset = MotionDataset(config["root_dir"], config["window_size"], mode ="train")
    val_dataset = MotionDataset(config["root_dir"], config["window_size"], mode ="val")
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=config["batch_size"],
                                              shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=config["batch_size"],
                                              shuffle=True)
   
    train_losses = []
    best_accuracy = 0

    
    
    wandb.config = {
    "epochs": config["num_epochs"],
    "batch_size": train_dataloader.batch_size,
    "learning_rate": config["learning_rate"],
    "loss_function": "Cross Entropy Loss",
    }


    for epoch in range(0, config['num_epochs']):
        model.train()
        epoch_losses = []
        
       
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch in train_pbar:
           
            data_sample = batch['data_sample'].to(device)
            class_label = batch['class_label'].to(device)
            
       
            optimizer.zero_grad()
            
            
            data_sample = data_sample.permute(0, 2, 1) 
            output = model(data_sample)
            
            
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
                }, os.path.join(config['best_dir'], 'best_model.pth'))

                print(f"Saved best model with accuracy: {accuracy:.2f}")
          
        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth'))
    
    wandb.finish()

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
            
            output_labels = model(data)
            _, outputs = torch.max(output_labels, dim = 1)

            total += labels.size(0)
            correct += (outputs == labels).sum().item()
    
    # Calculate accuracy
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
        model = TestCNN(input_channels=config['input_channels'],
        num_blocks=config['num_blocks'],
        num_features=config['num_features'])
        checkpoint = torch.load(os.path.join(config['best_dir'], 'best_model.pth'))


        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {config['best_dir']}")

    print("Testing")
    model.eval()
    
    correct = 0.0
    total = 0.0

    test_dataset =  MotionDataset(config["root_dir"], config["window_size"], mode ="test")

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
    
    # Calculate accuracy
    accuracy = 100 * correct / total

    classes = range(5)
    precision_list = []
    recall_list = []

    # find precision and recall statistics
    for i in range(5):
        precision = (precision_recall[i, 3]/ (precision_recall[i, 3] + precision_recall[i, 0])) * 100
        recall = (precision_recall[i, 3]/ (precision_recall[i, 3] + precision_recall[i, 1])) * 100
        precision_list.append(precision)
        recall_list.append(recall)
        print(f'Precision on Class {i} validation images: {precision:.2f}%')
        print(f'Recall on Class {i} validation images: {recall:.2f}%')


    plt.plot(classes, precision_list)
    plt.plot(classes, recall_list)
    plt.xlabel("Class Label")
    plt.ylabel("Percentage")
    plt.title("Precision and Recall Curve for Each Class")
    plt.savefig("Precision and Recall Curve for Each Class")

    print(f'Accuracy on the {total} validation images: {accuracy:.2f}%')
    
    

    
    return accuracy



if __name__ == "__main__":
    # Configuration
    config = {
        # Model parameters
        'num_features': 32,             # Number of feature channels
        'num_blocks': 10,               # Number of residual blocks
        

        'root_dir': fr'~\\DeepLearningFallDetection\\data',  # Training data directory
        'window_size' : 128,
        
        # Training parameters
        'batch_size': 10,                # Batch size
        'num_epochs': 50,               # Total number of epochs
        'learning_rate': 1e-5,           # Initial learning rate
        'lr_decay_step': 30,             # Epoch interval to decay LR
        'lr_decay_gamma': 0.5,           # Multiplicative factor of learning rate decay
        'validation_interval': 5,        # Epoch interval to perform validation (set to 5 for faster training)
        'input_channels'   : 6,
        

        'checkpoint_dir': 'checkpoints', # Directory to save checkpoints
        'run_type' : 'train',
        'best_dir': 'best_model',         # Directory to save sample images
        'save_every': 5,                 # Save checkpoint every N epochs
        'resume': None,                  # Path to checkpoint to resume from
    }
    

    if config['run_type'] == 'test':
        test()
    else:
        train(config)