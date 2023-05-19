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
import timeit

startTime = timeit.default_timer()

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (device)


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
        
        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return(loss_fn(pred, y).item())
    

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    #print (f'length of val loader {num_batches}')
    test_loss, correct = 0, 0
    prediction = []
    target = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)            
            pred = model(X)            
            test_loss += loss_fn(pred, y).item()
            #print (f'test_loss {test_loss}, pred {pred}, label {y}')
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            prediction.append(pred.argmax(1))
            target.append(y)

    test_loss /= num_batches
    correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, 100*correct, prediction, target

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
            ranked.append(0)
        #transition
        elif Y[i] > bin_edges[0] and Y[i] <= bin_edges[1]:
            ranked.append(1)
        #high
        elif Y[i] > bin_edges[1] and Y[i] <= bin_edges[2]:
            ranked.append(2)
        #Very High
        elif Y[i] > bin_edges[2]:
            ranked.append(3)
    
    return np.asarray(ranked)


def analyzeImages () :
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

    #print (X_train.shape, y_train.shape, X_test.shape)

    # resize binned_y to make it a 2D matrix
    y_train = y_train.reshape(-1,1)
    #print (y_train.shape)

    # Assign Ordinal labels 
    binned_y = rankbin (y_train)

    #-------------------------------------------------------------

    #----------Split Training Data-------------------------------
    train_in, val_in, train_y, val_y = train_test_split(X_train, binned_y, test_size=0.20)

    print (f' train {train_in.shape}, train_label {train_y.shape}, val {val_in.shape}, val label {val_y.shape}')
    #print (f' val_labels {val_y}')
    #------------------------------------------------------------

    #----------Define Image augmentation to increase number of images----------------
    # Define data transformations
    transform_train = transforms.Compose([
    #transforms.RandomCrop(30, padding = None),
    transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation((0, 359)),    
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),    
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    #-----------Preprocess the data-----------------------------------------------

    # Set batch_size for training and val
    batch_size_train = 20
    batch_size_val = 20

    # Create instances for training and validation datasets
    train_dataset = MyDataset(train_in, train_y, transform=transform_train)
    val_dataset = MyDataset(val_in, val_y, transform=transform_test)

    #print (f' val_dataset {val_dataset.targets}')
    #print (train_dataset.data.shape)
    # Create DataLoader for batching, shuffling
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                               batch_size = batch_size_train,
                                               shuffle = True)

    val_loader = torch.utils.data.DataLoader (dataset = val_dataset, batch_size=batch_size_val, shuffle=False)

    print (train_dataset.data.shape, val_dataset.data.shape)
   
    # debug prints

    #for data, targets in val_loader:
        #print(f'data_loader targets {targets}')
    #------------- Setup the Model------------------------------------------------
    
    # Number of outputs are 4 here becasuse there are 4 ordinal labels
    num_outputs = 4
    model = AlexNetBN(num_classes=num_outputs).to(device)

    #model.float()
    # initialize weights, requires forward pass for Lazy layers
    X = next(iter(train_loader))[0].to(device)    # get a batch from dataloader
    model.forward(X)                       # apply forward pass
    model.apply(init_weights)              # apply initialization

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 10

    plt.figure(figsize=(8,6))
    train_losses = []
    test_losses = []
    test_accs = []
    
    # the following lists will be overwritten for each epoch but i only care about the last one
    pred = []
    labels = []
    for t in range(epochs):
        train_loss = train_loop(train_loader, model, loss_fn, optimizer)
        val_loss, val_acc, pred, labels = test_loop(val_loader, model, loss_fn)

        # plot
        train_losses.append(train_loss)
        test_losses.append(val_loss)
        test_accs.append(val_acc/100)
        plt.clf()
        plt.plot(np.arange(1, t+2), train_losses, '-', label='train loss')
        plt.plot(np.arange(1, t+2), test_losses, '--', label='test loss')
        plt.plot(np.arange(1, t+2), test_accs, '-.', label='test acc')
        plt.legend()    
        display.clear_output(wait=True)
        display.display(plt.gcf())
        time.sleep(0.0001)
        
        pred_cpu = np.concatenate([p.cpu().numpy().flatten() for p in pred]) + 1
        labels_cpu = np.concatenate([l.cpu().numpy().flatten() for l in labels]) + 1
        #print (f'pred {pred_cpu.shape}, labels_cpu {labels_cpu.shape}')
        
        df = pd.DataFrame({
            'pred': pred_cpu, 
            'labels': labels_cpu
        })
        # Save the DataFrame as a CSV file
        df.to_csv('../results/predictions_labels.csv', index=False)

        # Calculate mean ordinal loss
        L1_loss = np.mean(np.abs (labels_cpu - pred_cpu))
        mean_L1_loss = np.round(L1_loss,2)

    print(f"Final Accuracy: {(val_acc):>0.1f}% , Mean Ordinal Loss {mean_L1_loss}")
    #plt.show()

    return val_acc, mean_L1_loss

def main ():

    Iter = 10  
    acc = []
    l1_loss = []

    for i in range (Iter):
        val_acc = 0.0
        mean_l1_loss = 0.0
        val_acc, mean_l1_loss = analyzeImages()

        acc.append (val_acc)
        l1_loss.append (mean_l1_loss)
    
    acc_np = np.asarray (acc)
    l1_loss_np = np.asarray (l1_loss)

    avg_acc = np.mean (acc_np)
    avg_loss = np.mean (l1_loss_np)

    print (f'Average Ordinal accuracy {np.round(avg_acc,2)} , average L1_loss {np.round(avg_loss,2)}')

if __name__ == '__main__':
   main()