import tenseal as ts
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm  # Make sure to import tqdm

class ConvNet(torch.nn.Module):
    def __init__(self, hidden=128, output=2):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=7, padding=0, stride=3)
        output_dim = (28 - 7) // 3 + 1
        output_size = output_dim * output_dim * 8
        self.fc1 = torch.nn.Linear(output_size, hidden)
        self.fc2 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        x = self.conv1(x)
        x = x.pow(2) 
        output_dim = (28 - 7) // 3 + 1
        output_size = output_dim * output_dim * 8
        x = x.view(-1, output_size) 
        x = self.fc1(x)
        x = x.pow(2)  # Squaring activation again
        x = self.fc2(x)
        return x
    
class EncConvNet:
    def __init__(self, torch_nn):
        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.kernel_size[0],
            torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
        
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()
        
        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()
        
    def forward(self, enc_x, windows_nb):
        # conv layer
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        
        # pack all channels into a single flattened vector
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        # square activation
        enc_x.square_()
        # fc1 layer
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        # square activation
        enc_x.square_()
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        return enc_x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

def enc_test(context, model, test_loader, criterion, kernel_shape, stride):
    test_loss = 0.0
    class_correct = list(0. for i in range(2))  # Assume 2 classes
    class_total = list(0. for i in range(2))
    
    # Counter initialization
    cnt = 0
    
    # Modification here: Adding tqdm around test_loader for progress bar
    for data, target in tqdm(test_loader, desc='Testing progress', total=min(250, len(test_loader))):
        # Encoding and encryption
        x_enc, windows_nb = ts.im2col_encoding(
            context, data.view(28, 28).tolist(), kernel_shape[0],
            kernel_shape[1], stride
        )
        # Encrypted evaluation
        enc_output = model(x_enc, windows_nb)
        # Decryption of result
        output = enc_output.decrypt()
        output = torch.tensor(output).view(1, -1)

        # compute loss
        loss = criterion(output, torch.tensor([target]))  # Make sure target is tensor and correct shape
        test_loss += loss.item()
        
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(torch.tensor([target]).data.view_as(pred)))
        # calculate test accuracy for each object class
        label = target
        class_correct[label] += correct.item()
        class_total[label] += 1
        cnt += 1
        if cnt == 100:
            break

    test_loss = test_loss / sum(class_total)
    print(f'Test Loss: {test_loss:.6f}\n')
    
    for i in range(len(class_correct)):
        print(f'Test Accuracy of class {i}: {100 * class_correct[i] / class_total[i]:.2f}% ({class_correct[i]}/{class_total[i]})')

    overall_accuracy = 100 * sum(class_correct) / sum(class_total)
    print(f'\nOverall Test Accuracy: {overall_accuracy:.2f}% ({sum(class_correct)}/{sum(class_total)})')
    return overall_accuracy

def test(model, test_loader, criterion):
    test_loss = 0.0
    class_correct = list(0. for i in range(2))  # Make sure this matches the number of classes (2 for dogs and cats)
    class_total = list(0. for i in range(2))

    model.eval()

    for data, target in test_loader:
        output = model(data)
        
        loss = criterion(output, target)
        test_loss += loss.item()

        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))

        for i in range(len(target)):
            label = target.data[i].item()  # Use .item() to get a scalar
            class_correct[label] += correct.item() if correct.dim() == 0 else correct[i].item()
            class_total[label] += 1

    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.6f}\n')

    for i in range(len(class_correct)):
        if class_total[i] > 0:
            print(f'Test Accuracy of {i}: {100 * class_correct[i] / class_total[i]:.2f}% ({class_correct[i]}/{class_total[i]})')
        else:
            print(f'Test Accuracy of {i}: N/A (no samples)')

    print(f'\nTest Accuracy (Overall): {100 * sum(class_correct) / sum(class_total):.2f}% ({sum(class_correct)}/{sum(class_total)})')

def train(model, train_loader, criterion, optimizer, n_epochs):
    # model in training mode
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

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    
    # model in evaluation mode
    model.eval()
    return model

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28,28)),
        transforms.ToTensor(),
    ])

    # Load dogs and cats data
    train_data = datasets.ImageFolder(root='your_path_of_training_set', transform=transform)
    test_data = datasets.ImageFolder(root='your_path_of_test_set', transform=transform)
    print(train_data.class_to_idx)
    # Create DataLoaders
    batch_size = 512
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    model = ConvNet()
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    
    model = train(model, train_loader, criterion, optimizer, 5)
    test(model, test_loader, criterion)


    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
    # required for encoding
    kernel_shape = model.conv1.kernel_size
    stride = model.conv1.stride[0]


    ## Encryption Parameters

    # controls precision of the fractional part
    bits_scale = 26

    # Create TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,  
        coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
    )

    # set the scale
    context.global_scale = pow(2, bits_scale)

    # galois keys are required to do ciphertext rotations
    context.generate_galois_keys()

    enc_model = EncConvNet(model)
    print("Entering enc_test")
    start = time.time()
    enc_test(context, enc_model, test_loader, criterion, kernel_shape, stride)
    end = time.time()
    print("model testing takes",end - start,"s")
    
    model_path = "complete_model.pth"
    torch.save(model, model_path)
