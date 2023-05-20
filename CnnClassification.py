import time
import numpy as np
import matplotlib.pyplot as plt
import h5py
import numpy as np
from matplotlib.patches import Rectangle
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
import PIL
from pytorch_grad_cam import GradCAM
#from pytorch_grad_cam.utils.image import show_cam_on_image,preprocess_image
import pandas as pd 
import optuna 
import timeit

startTime = timeit.default_timer()

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (device)

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    img = np.uint8(255 * cam)
    img_alpha = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img_alpha[:, :, 3] = 180  # This is the alpha value (0-255). Higher value means more transparency.
    return img_alpha

def relu_hook_function(module, grad_in, grad_out):
 
    #If there is a negative gradient, change it to zero
 
    # Get the corresponding input
    corresponding_input = grad_out[0]
    
    # Get the corresponding output
    corresponding_output = grad_in[0]
    
    # Zero out gradients where input and output is less than 0
    corresponding_input[corresponding_output < 0] = 0
    return (corresponding_input,)


def getData ():
    
    # Open training data
    file = h5py.File("../mp02files/DEM_train.h5", "r+")
    X_train = np.array(file["/images"])
    y_train = np.array(file["/meta"])
    file.close()

    # Open test data
    file = h5py.File("../mp02files/DEM_test_features.h5", "r+")
    X_test = np.array(file["/images"])
    file.close()   

    # Delete images which has very little information
    indices_to_delete = [6,76,91,104,108,156]

    X_train = np.delete(X_train, indices_to_delete, axis=0)
    y_train = np.delete(y_train, indices_to_delete, axis=0)

    # resize binned_y to make it a 2D matrix
    y_train = y_train.reshape(-1,1)
    
    # Assign Ordinal labels 
    binned_y = rankbin (y_train)

    #Split Training Data
    train_in, val_in, train_y, val_y = train_test_split(X_train, binned_y, test_size=0.20)

    print (f' train {train_in.shape}, train_label {train_y.shape}, val {val_in.shape}, val label {val_y.shape}')

    return train_in, val_in, train_y, val_y    

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


def feature_maps (val_in, model):

    model.eval()

    for image_index in range(val_in.shape[0]):
        # Feature Maps
        # adding channel dimension
        X = torch.tensor(val_in[image_index, np.newaxis, :, :], dtype=torch.float32).to(device) 

        #print (model)
        #Conv1 is at index 0 and Conv2 is at index 4 and Conv3 is at index 8
        with torch.no_grad():
            feature_maps = model.net[0](X.to(device))   
            feature_map1 = model.net[4](feature_maps.to(device))              
        
        print(feature_maps.cpu().shape)

        fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
        for i in range(0,32):
            row, col = i//8, i%8
            ax[row][col].imshow(feature_maps[i].cpu())  
        plt.suptitle('Feature Map Conv1')  
        #plt.show()

        fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
        for i in range(0, 32):
            row, col = i//8, i%8
            ax[row][col].imshow(feature_map1[i].cpu())
        plt.suptitle('Feature Map Conv2')
        #plt.show()
                
        # Get the learned filters
        filters = model.net[0].weight.detach().cpu().numpy()
        filters1 = model.net[4].weight.detach().cpu().numpy()
        #print(filters.shape)

        # Plot the learned filters
        fig, axs = plt.subplots(8, 8, figsize=(16,8))
        fig.subplots_adjust(hspace=0.4)
        axs = axs.ravel()
        for i in range(filters.shape[0]):
            axs[i].imshow(((filters[i].transpose(1, 2, 0)) * 255).astype(np.uint8), vmin=0, vmax=255, cmap='gray')
            axs[i].axis('off')
        plt.suptitle('Learned Filters for Conv1, index0')
        plt.show()

        fig, axs = plt.subplots(8, 8, figsize=(16,8))
        fig.subplots_adjust(hspace=0.4)
        axs = axs.ravel()
        # The reason this plot is black and white because its showing only 
        # 1 channel due to shape[1]
        for i in range(filters1.shape[1]):
            axs[i].imshow((filters1[i][i]* 255).astype(np.uint8), vmin=0, vmax=255, cmap='gray')
            axs[i].axis('off')
        plt.suptitle('Learned Filters for Conv2, index4')
        plt.show()      


def cam_vis (val_loader, model):
  
    heatResid_ordinal_classes = [
        'low', 'transition', 'high', 'very high'
    ]
    # Define a function to preprocess the image
    def preprocess_image(img):
        img = img.unsqueeze(0).to(device) # add batch dimension and send to device
        return img

    # Extract a batch of images and labels from the test_dataloader
    images, labels = next(iter(val_loader))     

    # For CAM, we should pick the last conv layer from the model
    target_layers = [model.net[14]]

    # Move the model to CPU and set it to evaluation mode
    model = model.eval().to(device)

    # Create a GradCAM object
    gradcam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    for i in range(3):
        for j in range(3):
            input_image = images[i*3 + j]
            true_label = labels[i*3 + j]
            true_class = heatResid_ordinal_classes[true_label]

            # Move the input_image to the device
            input_image = input_image.to(device)

            # Get the prediction for the input_image
            output = model(preprocess_image(input_image))
            _, predicted_label = torch.max(output, 1)  # get the predicted class
            
            predicted_class = heatResid_ordinal_classes[predicted_label]

            # Check if prediction is correct
            correct_prediction = (true_label == predicted_label)

            # Generate the CAM mask
            cam_mask = gradcam(input_tensor=preprocess_image(input_image))

            # Convert the images to RGB format
            rgb_img = np.moveaxis(input_image.cpu().numpy(), 0, -1)
            rgb_img = rgb_img.astype(np.float32)
            # normalize the image
            rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))  

            cam_image = show_cam_on_image(rgb_img, cam_mask[0, :])

            # Display the CAM
            axs[i, j].imshow(cam_image)
            axs[i, j].set_title(f'True Label: {true_class}, Predicted Label: {predicted_class}')
            #axs[i, j].axis('off')

            # Turn off only the axis labels and ticks
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

            # Add a red or green border depending on the correctness of prediction
            for spine in axs[i, j].spines.values():
                if correct_prediction:
                    spine.set_edgecolor('green')
                else:
                    spine.set_edgecolor('red')
                spine.set_linewidth(5)

    plt.tight_layout()
    plt.show()


def guidedBackProp(val_loader, model):
    # Guided Back Propagation
    heatResid_ordinal_classes = ['low', 'transition', 'high', 'very high']

    test_model = model.train().to(device)

    # Set requires_grad = True for all parameters
    for param in test_model.parameters():
        param.requires_grad = True

    for i, module in enumerate(test_model.modules()):
        if isinstance(module, torch.nn.ReLU):
            module.register_full_backward_hook(relu_hook_function)

    # Create a figure with 3x6 subplots
    fig, axs = plt.subplots(3, 6, figsize=(18, 9))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    # Extract a batch of images and labels from the test_dataloader
    images, labels = next(iter(val_loader))

    for i in range(3):
        for j in range(3):
            # Select the image from the batch
            input_image = images[i*3 + j]

            if len(input_image.shape) == 3:
                # Add a dummy batch dimension
                input_tensor = torch.unsqueeze(input_image, 0)
            else:
                input_tensor = input_image

            # Make input_tensor a leaf tensor and move it to the device
            input_tensor = input_tensor.clone().detach().to(device)
            input_tensor.requires_grad_(True)

            # forward/inference
            test_model.zero_grad()
            output = test_model(input_tensor)

            # Backward pass
            output.backward(torch.ones(output.shape).to(device))

            # get the absolute value of the gradients and take the maximum along the color channels
            gb = input_tensor.grad.abs().max(dim=1)[0].cpu().numpy()

            gb = gb - gb.min()
            gb = gb / gb.max()  # normalize to 0-1

            # remove color channel if it's 1
            input_image = np.squeeze(input_image)  

            # Get the image label
            image_label = labels[i*3 + j].item()
            image_class = heatResid_ordinal_classes[image_label]

            # Get the predicted label
            predicted_label = output.argmax().item()
            predicted_class = heatResid_ordinal_classes[predicted_label]

            # plot original image
            axs[i, j*2].imshow(input_image)
            axs[i, j*2].set_title(f'Orig Label: {image_class}')
            axs[i, j*2].axis('off')

            # plot guided backpropagation
            gb = np.squeeze(gb) 
            axs[i, j*2 + 1].imshow(gb, cmap ='gray')
            axs[i, j*2 + 1].set_title(f'GBP Pred Label: {predicted_class}')
            axs[i, j*2 + 1].axis('off')

    plt.show()  
  


def analyzeImages (train_in, val_in, train_y, val_y, epochs, batch_size_train, batch_size_val, lr) :
    

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
    num_outputs = 4
    model = AlexNetBN(num_classes=num_outputs).to(device)

    #model.float()
    # initialize weights, requires forward pass for Lazy layers
    X = next(iter(train_loader))[0].to(device)    # get a batch from dataloader
    model.forward(X)                       # apply forward pass
    model.apply(init_weights)              # apply initialization

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    plt.figure(figsize=(8,6))
    train_losses = []
    test_losses = []
    test_accs = []
    
    # the following lists will be overwritten for each epoch but i only care about the last one    
    for t in range(epochs):
        pred = []
        labels = []

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

    feature_maps(val_in, model)
    cam_vis (val_loader, model)
    guidedBackProp (val_loader, model)


    return val_acc, mean_L1_loss


def main ():

    train_in, val_in, train_y, val_y = getData()    
   
    Iter = 1
    acc = []
    l1_loss = []
    epoch = 10
    batch_size_train = 20
    batch_size_val = 44
    lr = 0.1
    for i in range (Iter):
        val_acc = 0.0
        mean_l1_loss = 0.0
        val_acc, mean_l1_loss = analyzeImages(train_in, val_in, train_y, val_y, epoch, batch_size_train, batch_size_val, lr)
        acc.append (val_acc)
        l1_loss.append (mean_l1_loss)
    acc_np = np.asarray (acc)
    l1_loss_np = np.asarray (l1_loss)
    avg_acc = np.mean (acc_np)
    avg_loss = np.mean (l1_loss_np)
    print (f'Average Ordinal accuracy {np.round(avg_acc,2)} , average L1_loss {np.round(avg_loss,2)}')



if __name__ == '__main__':
   main()