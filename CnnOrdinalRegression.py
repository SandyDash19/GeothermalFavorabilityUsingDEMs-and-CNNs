import time
import numpy as np
import matplotlib.pyplot as plt
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython import display
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from AlexNetBN import AlexNetBN
from sklearn.model_selection import train_test_split
import torchcam
from torchcam.methods import GradCAM
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image,preprocess_image
import pandas as pd 
import optuna 
import timeit

"""
Considering regression approach here
Input to the CNN - Images
Labels - heat residual with a cap 
  if heat resid < 0 :
     heat_resid = 0
  elif heat resid > 300 :
     heat_resid = 300

CNN num_output classes = 1
Optimizer = Adams
Loss = Smooth L1 Loss

Bin the targets and bin the predictions
Calculate mean L1 loss

The code in this file is mostly same as CnnClassification.py and I could
create a flag in CnnClassfication.py to use it for regression but having a separate
file is cleaner.

"""
startTime = timeit.default_timer()

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (device)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

# Apply Transform to dataset in this class.
class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        #self.targets = torch.FloatTensor(targets) # Use this for regression 
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]          
        
        if self.transform:
            x = Image.fromarray(x)  # Convert numpy array to PIL.Image
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)    

def init_weights(module):
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    train_loss = 0.0 
    for batch, (X, y) in enumerate(dataloader):
        #print (X.shape, y.shape)
        X = X.to(device)
        y = y.to(device)        
        # Compute prediction and loss        
        pred = model(X)
        #print (y.shape, pred.shape)
        #print (f'pred {pred}, y {y}')
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        if batch % 20 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= len(dataloader)
    return(loss_fn(pred, y).item()), train_loss
    

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    #print (f'length of val loader {num_batches}')
    test_loss, correct = 0, 0   
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)            
            pred = model(X)            
            test_loss += loss_fn(pred, y).item()
            #print (f'test_loss {test_loss}, pred {pred}, label {y}')            
    
        test_loss /= num_batches

#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, pred, y

"""This function gets the input of predicted outputs from validation set,
and actual labels for val set. These inputs are already quantile transformed.
This function bins the predicted and actual into 4 bins of heat as commented below.
Since the inputs are transformed therefore the bin edges are also transformed before
comparison"""

def rankbin (Y):

    ranked = []
    bin_edges = np.array([25,50,200])
    bin_edges = bin_edges.reshape(-1,1)    
    
    # Ordinal labels are proj labels -1 because of error 
    # nll_loss_forward_reduce_cuda_kernel_2d: block: [0,0,0], 
    # thread: [0,0,0] Assertion `t >= 0 && t < n_classes` failed. 
    
    for i in range (len(Y)):
        # Low 
        if Y[i] <= bin_edges[0]:
            ranked.append(1)
        #transition
        elif Y[i] > bin_edges[0] and Y[i] <= bin_edges[1]:
            ranked.append(2)
        #high
        elif Y[i] > bin_edges[1] and Y[i] <= bin_edges[2]:
            ranked.append(3)
        #Very High
        elif Y[i] > bin_edges[2]:
            ranked.append(4)
    
    return np.asarray(ranked)


def analyzeImages (epochs, batch_size_train, batch_size_val, lr) :
    #---------- Get Data---------------------------------------
    # Open training data
    file = h5py.File("../mp02files/DEM_train.h5", "r+")
    X_train = np.array(file["/images"])
    y_train = np.array(file["/meta"])
    file.close()

    # Open test data
    file = h5py.File("../mp02files/DEM_test_features.h5", "r+")
    X_test = np.array(file["/images"])
    file.close()

    print (X_train.shape, y_train.shape, X_test.shape)
    #print (f'y_train_max {y_train.max()}, y_train_min {y_train.min()}')

    # Delete images which has very little information
    indices_to_delete = [6,76,91,104,108,156]

    X_train = np.delete(X_train, indices_to_delete, axis=0)
    y_train = np.delete(y_train, indices_to_delete, axis=0)

    # Cap the targets 
    y_train[y_train < 0] = 0
    y_train[y_train > 300] = 300    

    # resize binned_y to make it a 2D matrix
    y_train = y_train.reshape(-1,1)

    #print (y_train , y_train.shape)
    

    #-------------------------------------------------------------

    #----------Split Training Data-------------------------------
    train_in, val_in, train_y, val_y = train_test_split(X_train, y_train, test_size=0.20)

    print (f' train {train_in.shape}, train_label {train_y.shape}, val {val_in.shape}, val label {val_y.shape}')
    #print (f' val_labels {val_y}')
    #------------------------------------------------------------

    #----------Define Image augmentation to increase number of images----------------
    # Define data transformations
    transform_train = transforms.Compose([
    #transforms.RandomCrop(30, padding = None),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation((0, 359)),       
    #transforms.RandomPerspective(distortion_scale=0.5, p=0.5),    
    #transforms.GaussianBlur(3, sigma=(0.1, 2.0)),    
    transforms.ToTensor(),
    #AddGaussianNoise(0.0, 0.1),
    transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    #-----------Preprocess the data-----------------------------------------------

    # Create instances for training and validation datasets
    train_dataset = MyDataset(train_in, train_y, transform=transform_train)
    val_dataset = MyDataset(val_in, val_y, transform=transform_test)

    #print (f' val_dataset {val_dataset.targets}')
    #print (train_dataset.data.shape)
    # Create DataLoader for batching, shuffling
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                               batch_size = batch_size_train,
                                               shuffle = True)

    val_loader = torch.utils.data.DataLoader (dataset = val_dataset, batch_size=batch_size_val, shuffle=True)

    print (train_dataset.data.shape, val_dataset.data.shape)
   
    # debug prints

    #for data, targets in val_loader:
        #print(f'data_loader targets {targets}')
    #------------- Setup the Model------------------------------------------------
    
    # Number of outputs are 4 here becasuse there are 4 ordinal labels
    num_outputs = 1
    model = AlexNetBN(num_classes=num_outputs).to(device)

    #model.float()
    # initialize weights, requires forward pass for Lazy layers
    X = next(iter(train_loader))[0].to(device)    # get a batch from dataloader
    model.forward(X)                       # apply forward pass
    model.apply(init_weights)              # apply initialization

    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    plt.figure(figsize=(8,6))
    train_losses = []
    test_losses = []
    my_train_losses = []
    test_accs = []
      
   
    for t in range(epochs):

        pred = []
        labels = []

        train_loss, my_train_loss = train_loop(train_loader, model, loss_fn, optimizer)
        val_loss, pred, labels = test_loop(val_loader, model, loss_fn)

        # plot
        my_train_losses.append(my_train_loss)
        train_losses.append(train_loss)
        test_losses.append(val_loss)
       
        plt.clf()
        plt.plot(np.arange(1, t+2), my_train_losses, '-', label='my_train loss')
        #plt.plot(np.arange(1, t+2), train_losses, '-', label='train loss')
        plt.plot(np.arange(1, t+2), test_losses, '--', label='test loss')
        
        plt.legend()    
        display.clear_output(wait=True)
        display.display(plt.gcf())
        time.sleep(0.0001)
        
        pred_cpu = np.concatenate([p.cpu().numpy().flatten() for p in pred])
        labels_cpu = np.concatenate([l.cpu().numpy().flatten() for l in labels])
        
        #bin labels and prediction
        binned_pred = rankbin (pred_cpu)
        binned_labels = rankbin (labels_cpu)

    df = pd.DataFrame({
            'pred': binned_pred, 
            'labels': binned_labels
        })
    # Save the DataFrame as a CSV file
    df.to_csv('../results/predictions_labels_regression.csv', index=False)

    # Calculate mean ordinal loss
    L1_loss = np.mean(np.abs (binned_labels - binned_pred))
    mean_L1_loss = np.round(L1_loss,2)

    correct = 0.0
    for i in range (binned_pred.shape[0]):
            correct += (binned_pred[i] == binned_labels[i])

    val_acc = correct / binned_labels.shape[0]
    
    val_acc *= 100

    print(f"Final Accuracy: {(val_acc):>0.1f}% , Mean Ordinal Loss {mean_L1_loss}")
    plt.show()

    return val_acc, mean_L1_loss


def objective(trial):
    lr = trial.suggest_float('lr', 1e-3, 1e-1)
    batch_size_train = trial.suggest_categorical('batch_size_train', [8,16,24,32,40,48,56,64,72,80])
    batch_size_val = 44
    epochs = trial.suggest_int('epochs', 10, 100)

    val_acc, mean_l1_loss = analyzeImages(epochs, batch_size_train, batch_size_val, lr)
    return mean_l1_loss

def main ():

    optuna_enabled = 0
    
    if optuna_enabled:

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30)

        print('Best trial:')
        trial = study.best_trial
        print(f'  Value: {trial.value}')
        print('  Params: ')
        for key, value in trial.params.items():
            print(f'    {key}: {value}')
    
    else :
        Iter = 1
        acc = []
        l1_loss = []

        epoch = 66
        batch_size_train = 20
        batch_size_val = 44
        lr = 0.1

        for i in range (Iter):
            val_acc = 0.0
            mean_l1_loss = 0.0
            val_acc, mean_l1_loss = analyzeImages(epoch, batch_size_train, batch_size_val, lr)

            acc.append (val_acc)
            l1_loss.append (mean_l1_loss)

        acc_np = np.asarray (acc)
        l1_loss_np = np.asarray (l1_loss)

        avg_acc = np.mean (acc_np)
        avg_loss = np.mean (l1_loss_np)

        print (f'Average Ordinal accuracy {np.round(avg_acc,2)} , average L1_loss {np.round(avg_loss,2)}')


if __name__ == '__main__':
   main()