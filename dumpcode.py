# Working CAM function 

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

    # Select the first image from the batch
    input_image = images[0]
    true_label = labels[0]
    true_class = heatResid_ordinal_classes[true_label]

    # Move the model to CPU and set it to evaluation mode
    model = model.eval().to(device)

    # Move the input_image to the device
    input_image = input_image.to(device)

    # Get the prediction for the input_image
    output = model(preprocess_image(input_image))
    _, predicted_label = torch.max(output, 1)  # get the predicted class
    predicted_class = heatResid_ordinal_classes[predicted_label]

    # For CAM, we should pick the last conv layer from the model
    target_layers = [model.net[14]]

    # Create a GradCAM object
    gradcam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    # Generate the CAM mask
    cam_mask = gradcam(input_tensor=preprocess_image(input_image))

	# Convert the images to RGB format
    rgb_img = np.moveaxis(input_image.cpu().numpy(), 0, -1)
    rgb_img = rgb_img.astype(np.float32)
	# normalize the image
    rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))  


    #print (rgb_img.shape, cam_mask.shape)
    
    cam_image = show_cam_on_image(rgb_img, cam_mask[0, :])

    # Display the CAM
    plt.imshow(cam_image)
    plt.title(f'True Label: {true_class}, Predicted Label: {predicted_class}')
    plt.show()


def guidedBackProp (val_loader, model):

    # Guided Back Propagation
    heatResid_ordinal_classes = [
        'low', 'transition', 'high', 'very high'
    ]

    test_model = model.train().to(device)

    # Set requires_grad = True for all parameters
    for param in test_model.parameters():
        param.requires_grad = True

    for i, module in enumerate(test_model.modules()):
        if isinstance(module, torch.nn.ReLU):
            #print(test_model.named_modules())
            module.register_full_backward_hook(relu_hook_function)


    # Extract a batch of images and labels from the test_dataloader
    images, labels = next(iter(val_loader))

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

    gb = gb - gb.min()
    gb = gb / gb.max()  # normalize to 0-1

    plt.figure(figsize=(6, 6))

    # plot original image
    plt.subplot(1, 2, 1)

    # remove color channel if it's 1
    input_image = np.squeeze(input_image)  

    # Get the image label
    image_label = labels[0].item()
    image_class = heatResid_ordinal_classes[image_label]

    # Get the predicted label
    predicted_label = output.argmax().item()
    predicted_class = heatResid_ordinal_classes[predicted_label]

    plt.imshow(input_image)
    plt.title(f'Orig Label: {image_class}')
    plt.axis('off')

    # plot guided backpropagation
    plt.subplot(1, 2, 2)  
    # remove the unnecessary channel dimension 
    gb = np.squeeze(gb) 
    plt.imshow(gb, cmap ='cividis')
    plt.title(f'GBP Pred Label: {predicted_class}')
    plt.axis('off')

    plt.show()  

 