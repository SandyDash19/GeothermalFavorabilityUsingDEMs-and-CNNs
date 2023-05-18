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
import os
import timeit

startTime = timeit.default_timer()

def rankbin (Y):

    ranked = []
    bin_edges = np.array([25,50,200])
    bin_edges = bin_edges.reshape(-1,1)   

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
print (f'y_train_max {y_train.max()}, y_train_min {y_train.min()}')

# Bin the labels 
binned_y = rankbin(y_train)

# resize binned_y to make it a 2D matrix
binned_y = binned_y.reshape(-1,1)
#print (binned_y.shape)

# initialize norm_y
norm_y = np.zeros_like(binned_y, dtype=np.float32)

# normalize binned_y
min_val = binned_y.min()
max_val = binned_y.max()
range_val = max_val - min_val

for i in range(binned_y.shape[0]):
    norm_y[i] = (binned_y[i] - min_val) / range_val

#print (norm_y)

norm_y = norm_y.astype(np.float16)
#np.savetxt('Labels_binned.csv', binned_y, delimiter=',')
#-------------------------------------------------------------

#----------Split Training Data-------------------------------
train_in, val_in, train_y, val_y = train_test_split(X_train, binned_y, test_size=0.20)

print (train_in.shape, train_y.shape, val_in.shape, val_y.shape)
#------------------------------------------------------------

#----------Perform Image augmentation to increase number of images----------------
# Define data transformations
transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914], std=[0.2023])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914], std=[0.2023])
])

#---------------------------------------------------------------------------------

#--------- Apply image augmentation to train and val-----------------------------

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.FloatTensor(targets)
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

# download and preprocess data, create dataloaders
batch_size_train = 8
batch_size_val = 1

# Create instances for training and validation datasets
train_dataset = MyDataset(train_in, train_y, transform=transform_train)
val_dataset = MyDataset(val_in, val_y, transform=transform_test)

print (train_dataset.data.shape)
# Create DataLoader for batching, shuffling
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size_train,
                                           shuffle = True)

val_loader = torch.utils.data.DataLoader (dataset = val_dataset, batch_size=batch_size_val, shuffle=False)

print (train_dataset.data.shape, val_dataset.data.shape)
#--------------------------------------------------------------------------------

#-----------Train and Test Loops -----------------------------------------------
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
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            print (f' pred = {pred} , target = {y}')
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()            

    test_loss /= num_batches
    correct /= size
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, 100*correct
#-----------------------------------------------------------------------------

#------------- Setup the Model------------------------------------------------

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (device)

# Number of outputs are 4 here becasuse 
num_outputs = 4
model = AlexNetBN(num_outputs).to(device)

model.float()
# initialize weights, requires forward pass for Lazy layers
X = next(iter(train_loader))[0].to(device)    # get a batch from dataloader
model.forward(X)                       # apply forward pass
model.apply(init_weights)              # apply initialization

loss_fn = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 30

plt.figure(figsize=(8,6))
train_losses = []
test_losses = []
test_accs = []
for t in range(epochs):
    train_loss = train_loop(train_loader, model, loss_fn, optimizer)
    val_loss, val_acc = test_loop(val_loader, model, loss_fn)
    
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
    
print(f"Final Accuracy: {(val_acc):>0.1f}%")
plt.show()

#----------------------------------------------------------------------
# Save seems to happen but reload of dictionary is missing all the weights

"""torch.save({
    'model_state_dict': model.state_dict(),
}, 'AlexNet_cifar10.pth')"""

"""
# Feature Maps
X = torch.tensor ([train_dataset.data[7]], dtype=torch.float32).permute(0,3,1,2)

model.eval()

#print (model)
#Conv1 is at index 0 and Conv2 is at index 4 and Conv3 is at index 8
with torch.no_grad():
    feature_maps = model.net[0](X.to(device))   
    feature_map1 = model.net[4](feature_maps.to(device))   
    feature_map2 = model.net[8](feature_map1.to(device))   
   
fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
for i in range(0, 32):
    row, col = i//8, i%8
    ax[row][col].imshow(feature_maps[0][i].cpu())  
plt.suptitle('Feature Map Conv1, index0')  
plt.show()

fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
for i in range(0, 32):
    row, col = i//8, i%8
    ax[row][col].imshow(feature_map1[0][i].cpu())
plt.suptitle('Feature Map Conv2, index4')
plt.show()

fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
for i in range(0, 32):
    row, col = i//8, i%8
    ax[row][col].imshow(feature_map2[0][i].cpu())
plt.suptitle('Feature Map Conv3, index8')
plt.show()

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



# CAM code for CIFAR 10 images 
# Guided Back Propagation
cifar10_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
# Define a function to preprocess the image
def preprocess_image(img):
    img = img.unsqueeze(0).to(device) # add batch dimension and send to device
    return img

# Extract a batch of images and labels from the test_dataloader
images, labels = next(iter(test_dataloader))

# Select the first image from the batch
input_image = images[0]
true_label = labels[0]
true_class = cifar10_classes[true_label]

# Move the model to CPU and set it to evaluation mode
model = model.eval().to(device)

# Move the input_image to the device
input_image = input_image.to(device)

# Get the prediction for the input_image
output = model(preprocess_image(input_image))
_, predicted_label = torch.max(output, 1)  # get the predicted class
predicted_class = cifar10_classes[predicted_label]

# For CAM, we should pick the last conv layer from the model
target_layers = [model.net[10]]

# Create a GradCAM object
gradcam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

# Generate the CAM mask
cam_mask = gradcam(input_tensor=preprocess_image(input_image))

# Convert the images to RGB format
rgb_img = np.moveaxis(input_image.cpu().numpy(), 0, -1)
rgb_img = rgb_img.astype(np.float32)
# normalize the image
rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))  


# Resize the image
desired_size = (32, 32)  # specify the desired size
rgb_img = cv2.resize(rgb_img, desired_size, interpolation=cv2.INTER_CUBIC)

cam_image = show_cam_on_image(rgb_img, cam_mask[0, :])

# Display the CAM
plt.imshow(cam_image)
plt.title(f'True Label: {true_class}, Predicted Label: {predicted_class}')
plt.show()



# Guided Back Propagation
cifar10_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

test_model = model.train().to(device)

# Set requires_grad = True for all parameters
for param in test_model.parameters():
    param.requires_grad = True

#for param in test_model.parameters():
#    print(param.requires_grad)


def relu_hook_function(module, grad_in, grad_out):
    if isinstance(module, nn.ReLU):
        return (torch.clamp(grad_in[0], min=0.0),)


for i, module in enumerate(test_model.modules()):
    if isinstance(module, torch.nn.ReLU):
        #print(test_model.named_modules())
        module.register_full_backward_hook(relu_hook_function)


# Extract a batch of images and labels from the test_dataloader
images, labels = next(iter(test_dataloader))

# Select the first image from the batch
input_image = images[0]

#print (input_image.shape)

if len(input_image.shape) == 3:
    # Add a dummy batch dimension
    input_tensor = torch.unsqueeze(input_image, 0)
else:
    input_tensor = input_image

#print (input_tensor.shape)

# Make input_tensor a leaf tensor and move it to the device
input_tensor = input_tensor.clone().detach().to(device)
input_tensor.requires_grad_(True)

#print(f"Is input_tensor a leaf tensor? {input_tensor.is_leaf}") 

# forward/inference
test_model.zero_grad()
output = test_model(input_tensor)

# Backward pass
output.backward(torch.ones(output.shape).to(device))

#print(input_tensor.grad)  # Should not be None

# get the absolute value of the gradients and take the maximum along the color channels
gb = input_tensor.grad.abs().max(dim=1)[0].cpu().numpy()

plt.figure(figsize=(6, 6))

# plot original image
plt.subplot(1, 2, 1)

input_image = np.transpose(input_image, (1, 2, 0))

# Get the image label
image_label = labels[0].item()
image_class = cifar10_classes[image_label]

# Get the predicted label
predicted_label = output.argmax().item()
predicted_class = cifar10_classes[predicted_label]

plt.imshow(input_image)
plt.title(f'Orig Label: {image_class}')
plt.axis('off')

# plot guided backpropagation
plt.subplot(1, 2, 2)
gb = np.transpose(gb, (1,2,0))
plt.imshow(gb, cmap='hot')
plt.title(f'GBP Pred Label: {predicted_class}')
plt.axis('off')

plt.show()

"""


endTime = timeit.default_timer()
#total time 
print (f'Runtime of the program {(endTime - startTime)/60} minutes') 

