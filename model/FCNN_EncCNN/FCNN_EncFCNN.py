import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import time
from torch.utils.data import DataLoader
import tenseal as ts
import os
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

torch.manual_seed(73)

# Parameters
batch_size = 128
pixels = 28
input_size = pixels * pixels
mean = [0.4584]
std = [0.2492]
n_epochs = 50

# Paths to saved model state
model_full_path_plain = ""

# Paths to data folders
train_dir = 'your_path_to_training_set'
test_dir = 'your_path_to_test_set'

# Data transformation
transform = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.Resize((pixels, pixels)),  
    transforms.ToTensor(),  # Convert to torch.Tensor and normalize [0, 1]
    transforms.Normalize(mean, std),
])

# Load data
train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

# DataLoader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

print('{0: "Cat", 1: "Dog"}')

class FullyConnectedNet(torch.nn.Module):
    def __init__(self, input_size=pixels*pixels, hidden=64, output=2):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 1024)
        self.fc2 = torch.nn.Linear(1024, hidden)
        self.fc3 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        x = x.view(-1, input_size)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def test(model, test_loader, criterion):
    test_loss = 0.0
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))

    # model in evaluation mode
    model.eval()

    for data, target in test_loader:
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = pred.eq(target.data.view_as(pred))
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {test_loss:.6f}\n')

    # Print accuracy for each class
    for label in range(2):
        if class_total[label] > 0:
            print(
                f'Test Accuracy of {label} ({labels_map[label]}): {100 * class_correct[label] / class_total[label]:.2f}% '
                f'({int(class_correct[label])}/{int(class_total[label])})'
            )
        else:
            print(f'Test Accuracy of {label} ({labels_map[label]}): N/A (no samples)')
            
    print(f'\nOverall Test Accuracy: {accuracy:.2f}%\n')

    return test_loss, accuracy

def train(model, train_loader, criterion, optimizer,scheduler, n_epochs):
    # Initialize lists to save metrics at each epoch
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_accuracies = []

    model.train()
    for epoch in range(1, n_epochs+1):

        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # calculate average losses
        train_loss = train_loss / len(train_loader)
        epoch_train_losses.append(train_loss)

        # Validation phase
        val_loss, val_accuracy = test(model, test_loader, criterion)
        epoch_val_losses.append(val_loss)
        epoch_accuracies.append(val_accuracy)

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
        scheduler.step()
    # model in evaluation mode
    model.eval()
    return model, epoch_train_losses, epoch_val_losses, epoch_accuracies

model = FullyConnectedNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# Check if a pre-trained model state exists and load it
if os.path.exists(model_full_path_plain):
    print("Loading pre-trained plain model...")
    model = torch.load(model_full_path_plain)
else:
    print("Training a new model...")

# Whether the model is loaded or newly initialized, we can continue training here
print("Training the plain model...")
model.train()  # Make sure the model is in training mode
start = time.time()
model, epoch_train_losses, epoch_val_losses, epoch_accuracies = train(model, train_loader, criterion, optimizer,scheduler, n_epochs)  
end = time.time()
print("Plain model training took", end - start, "seconds")

torch.save(model, model_full_path_plain)               

# Test the plain model
start = time.time()
test(model, test_loader, criterion)
end = time.time()
print("Model testing took",end - start,"s\n")

class EncFullyConnectedNet: 
    def __init__(self, torch_nn):
        # Initialize with weights and biases of plain model converted to lists
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()
        
        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()
        
        self.fc3_weight = torch_nn.fc3.weight.T.data.tolist()
        self.fc3_bias = torch_nn.fc3.bias.data.tolist()
        
        
    def forward(self, enc_x):
        # Evaluate the model on encrypted data
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        enc_x.square_()
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        enc_x.square_()
        enc_x = enc_x.mm(self.fc3_weight) + self.fc3_bias
        return enc_x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# Define the size of the test subset
n_test_samples = 100  

# Create random indices for the test subset
test_indices = np.random.choice(len(test_data), n_test_samples, replace=False)

# Create a DataLoader for the test subset
test_sampler = SubsetRandomSampler(test_indices)
test_subset_loader = DataLoader(test_data, batch_size=1, sampler=test_sampler)

def enc_test(context, model, test_loader, criterion):
    times = []  # List to store computation times for each sample

    test_loss = 0.0
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    class_counts = {0: 0, 1: 0}  # For debugging: count number of images per class
    sample_counter = 0

    for data, target in test_loader:
        sample_counter += 1  # Increment the counter for each processed sample
        # Flatten data properly for each image
        data_flat = data.view(-1).tolist()  # Removes all dimensions except the features dimension
        # Measure encryption time
        start_time = time.time()
        # Encrypt data
        x_enc = ts.ckks_vector(context, data_flat)
        encryption_time = time.time() - start_time
        # Perform prediction
        # Measure inference time
        start_time = time.time()
        enc_output = model(x_enc)
        inference_time = time.time() - start_time
        # Measure decryption time
        start_time = time.time()
        output = enc_output.decrypt()
        output = torch.tensor(output).view(1, -1)
        decryption_time = time.time() - start_time

        # Add total time for this sample
        times.append(encryption_time + inference_time + decryption_time)

        # Adjust target dimension for loss function
        target = target.view(-1)  # Removes batch dimension if present
        # Calculate loss
        loss = criterion(output, target)
        test_loss += loss.item()
        # Determine predicted class
        _, pred = torch.max(output, 1)
        correct = pred.eq(target.view_as(pred))
        label = target.item()
        class_correct[label] += correct.item()
        class_total[label] += 1
        class_counts[label] += 1

        # Display after each prediction
        print(f"Sample #{sample_counter}: target={label}, prediction={pred.item()}")

    # Calculate and display average test loss and accuracy for each class
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.6f}\n')

    # Display accuracy for each class
    labels_map = {0: "Cat", 1: "Dog"}
    for label in range(2):
        if class_total[label] > 0:
            print(f'Test Accuracy of {labels_map[label]} (class {label}): {100 * class_correct[label] / class_total[label]:.2f}% ({class_correct[label]}/{class_total[label]})')
        else:
            print(f'Test Accuracy of {labels_map[label]} (class {label}): N/A (no samples)')

    # Display distribution of classes in the tested images
    print("Distribution of classes in tested images:", class_counts)
    # Calculate average computation time
    average_time = sum(times) / len(times)
    print(f"Average computation time per sample: {average_time:.4f} seconds")
    
    return times  

## Encryption Parameters

# Controls precision of the fractional part
bits_scale = 26

# Create TenSEAL context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
)

# Set the scale
context.global_scale = pow(2, bits_scale)

# Generate Galois keys required for ciphertext rotations
context.generate_galois_keys()

# Initialize encrypted model
enc_model = EncFullyConnectedNet(model)
start = time.time()
# Test encrypted model with the random subset
times = enc_test(context, enc_model, test_subset_loader, criterion)
end = time.time()
print("Encrypted model testing took",end - start,"seconds")
