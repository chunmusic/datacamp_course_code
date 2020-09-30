# Exercise_1 
# Create 10 random images of shape (1, 28, 28)
images = torch.rand(10, 1, 28, 28)

# Build 6 conv. filters
conv_filters = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)

# Convolve the image with the filters 
output_feature = conv_filters(images)
print(output_feature.shape)

--------------------------------------------------
# Exercise_2 
# Create 10 random images
image = torch.rand(10, 1, 28, 28)
# Create 6 filters
filters = torch.rand(6, 1, 3, 3)

# Convolve the image with the filters
output_feature = F.conv2d(image, filters, stride=1, padding=1)
print(output_feature.shape)

--------------------------------------------------
# Exercise_3 
# Build a pooling operator with size `2`.
max_pooling = torch.nn.MaxPool2d(2)

# Apply the pooling operator
output_feature = max_pooling(im)

# Use pooling operator in the image
output_feature_F = F.max_pool2d(im,2)

# print the results of both cases
print(output_feature)
print(output_feature_F)

--------------------------------------------------
# Exercise_4 
# Build a pooling operator with size `2`.
avg_pooling = torch.nn.AvgPool2d(2)

# Apply the pooling operator
output_feature = avg_pooling(im)

# Use pooling operator in the image
output_feature_F = F.avg_pool2d(im,2)

# print the results of both cases
print(output_feature)
print(output_feature_F)

--------------------------------------------------
# Exercise_5 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Instantiate two convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)
        
        # Instantiate the ReLU nonlinearity
        self.relu = nn.ReLU()
        
        # Instantiate a max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Instantiate a fully connected layer
        self.fc = nn.Linear(7 * 7 * 10, 10)

--------------------------------------------------
# Exercise_6 
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
		
        # Instantiate the ReLU nonlinearity
        self.relu = nn.ReLU()
        
        # Instantiate two convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)
        
        # Instantiate a max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Instantiate a fully connected layer
        self.fc = nn.Linear(7 * 7 * 10, 10)

    def forward(self, x):

        # Apply conv followd by relu, then in next line pool
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # Apply conv followd by relu, then in next line pool
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Prepare the image for the fully connected layer
        x = x.view(-1, 7 * 7 * 10)

        # Apply the fully connected layer and return the result
        return self.fc(x)

--------------------------------------------------
# Exercise_7 
for i, data in enumerate(train_loader, 0):
    inputs, labels = data
    optimizer.zero_grad()

    # Compute the forward pass
    outputs = net(inputs)
        
    # Compute the loss function
    loss = criterion(outputs,labels)
        
    # Compute the gradients
    loss.backward()
    
    # Update the weights
    optimizer.step()

--------------------------------------------------
# Exercise_8 
# Iterate over the data in the test_loader
for i, data in enumerate(test_loader):
  
    # Get the image and label from data
    image, label = data
    
    # Make a forward pass in the net with your image
    output = net(image)
    
    # Argmax the results of the net
    _, predicted = torch.max(output.data, 1)
    if predicted == label:
        print("Yipes, your net made the right prediction " + str(predicted))
    else:
        print("Your net prediction was " + str(predicted) + ", but the correct label is: " + str(label))

--------------------------------------------------
