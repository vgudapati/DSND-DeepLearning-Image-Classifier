# Imports

import time
import argparse
import numpy as np
import seaborn as sns
from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt

# Functions


def argparse_train():
    '''
    This function is to define the argument parser for training
    '''
    # Parser definition
    parser = argparse.ArgumentParser(description='Neural network training parameters')
    
    # Network architecture for training.
    parser.add_argument('--arch',
                        type=str,
                        help='Neural network architecture from torchvision.models'
                       )
    
    # Hyper parameters for the network
    
    parser.add_argument('--n_inputs',
                        type=int,
                        help='Number of input units for the classifier (integer)'
                       )
    
    parser.add_argument('--n_hidden',
                        type=int,
                        help='Number of hidden units for the classifier (integer)'
                       )
    
    parser.add_argument('--n_classes',
                        type=int,
                        help='Number of output units for training (integer)'
                       )
    parser.add_argument('--epochs',
                        type=int,
                        help='Number of epochs units for training (integer)'
                       )
    
    parser.add_argument('--learning_rate',
                        type=float,
                        help='Learning rate to be used for training (float)'
                       )
    
    parser.add_argument('--batch_size',
                        type=int,
                        help='Batch size to be used for training (integer)'
                       )
    
    parser.add_argument('--print_every',
                        type=int,
                        help='Print every few steps during training (integer)'
                       )
    
    parser.add_argument('--drop_prob',
                        type=float,
                        help='Drop probabiliry for regularization during training (float)'
                       )
    
    
    # GPU option
    parser.add_argument('--gpu',
                        action='store_true',
                        help='Use GPU for training or not'
                       )
    
    parser.add_argument('--checkpoint',
                        type=str,
                        help='check point file name (string)'
                       )
    
    return parser.parse_args()
    
def train_transformer(train_dir):
    '''
    Function to perform transformations on the training data set
    '''
    
    # Defines the RandomRotation, RandomResizedCrop, RandomHorizontalFlipm, Normalize 
    # transformations on training data set
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    
    # Load the data set with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data

def test_transformer(test_dir):
    '''
    Function to perform transformations on the training data set
    '''
    
    # Defines the Resize, CenterCrop, Normalize, transformations on test data set
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    # Load the data set with ImageFolder
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data
    
def data_loader(data, batch_size, train=True):
    '''
    Function to load data for each set with DataLoader
    '''
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    return loader

def check_gpu(gpu_arg):
    '''
    Functions to check the device(GPU or CPU) used for training.
    '''
    # If requested to use GPU, check for it and if available, use it else use CPU
    if gpu_arg:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            print("CUDA was not found on device, using CPU instead.")
    else:
        device = torch.device("cpu")
    
    return device

def build_model(arch, n_inputs, n_hidden, n_classes, drop_prob):
    '''
    Function to create the initial pretrained model for training.
    '''
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif arch == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print('Model architecture provided is not suppored')
        
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(n_inputs, n_hidden),
                       nn.ReLU(),
                       nn.Dropout(p=drop_prob),
                       nn.Linear(n_hidden, n_classes),
                       nn.LogSoftmax(dim=1))
    
    model.classifier = classifier
    
    return model

def validation(model, loader, criterion, device):
    '''
    Function to validate the training against the test loader and return loss and accuracy
    '''
    accuracy = 0
    loss = 0
    model.eval()
    
    for inputs, labels in loader:

        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model.forward(inputs)
        loss += criterion(outputs, labels).item()

        ps = torch.exp(outputs)
        equality = (labels.data == ps.max(1)[1])
        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

    return loss, accuracy
    
def train(model, trainloader, validloader, criterion, optimizer, device, epochs=1, print_every=40):
    '''
    Function to trainthe neural network
    '''
    print_every = print_every
    running_loss = 0
    valid_loss = 0
    accuracy = 0
    stepend = 0
    
    model.train()
    model.to(device)
    training_loss_per_epoch = 0
    #print('in train')
    #print(epochs)
    for e in range(epochs):
        
        steps = 0
        epochstart = time.time()
        # Model in training mode, dropout is on
        model.train()
        for inputs, labels in trainloader:
            stepstart = time.time()
            steps += 1
            #print(steps)
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if print_every:
                if steps % print_every == 0:
                    # Model in inference mode, dropout is off
                    model.eval()

                    with torch.no_grad():
                        valid_loss, accuracy = validation(model, validloader, criterion, device)



                    stepend = time.time()
                    #print(print_every, 'steps took', stepend - stepstart, 'seconds', end='\n')
                    print("Epoch: {}/{} |".format(e+1, epochs),
                          "Steps: {:3d}[{:7.3f}{}] |".format(steps, stepend - epochstart, 's'),
                          "Training Loss: {:.3f} |".format(running_loss/print_every),
                          "Validation Loss: {:.3f} |".format(valid_loss/len(validloader)),
                          "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                    training_loss_per_epoch += running_loss
                    running_loss = 0
                    # Make sure dropout and grads are on for training
                    model.train()
                
        epochend = time.time()
    return model
    
def test_model(model, loader, device):
    '''
    Function to test the model accuracy
    '''
    # Do validation on the test set
    correct = 0
    total = 0
    model.eval()
    if device != 'cpu':
        model.cuda()
    for data in loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print('Accuracy achieved by the network on test images is: %d%%' % (100 * correct / total))

def save_model(model, path, arch, hidden, class_to_idx, optimizer):
    '''
    Function to save the model
    '''
    model.class_to_idx = class_to_idx
    model.eval()
    checkpoint = {'arch': arch,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'hidden':hidden,
                  'inputs':inputs}
    
    torch.save(checkpoint, path)

def main():
    
    # Get the args for training
    args = argparse_train()
    
    # Data location
    data_dir = 'flower_data'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Data transformers and Data loaders
    train_data = train_transformer(train_dir)
    valid_data = test_transformer(valid_dir)
    test_data = test_transformer(test_dir)
    
    trainloader = data_loader(train_data, batch_size=args.batch_size)
    validloader = data_loader(valid_data, batch_size=args.batch_size, train=False)
    testloader = data_loader(test_data, batch_size=args.batch_size, train=False)
    
    model = build_model(args.arch, args.n_inputs, args.n_hidden, args.n_classes, args.drop_prob)
    device = check_gpu(gpu_arg = args.gpu)
    print('Using', device)
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    trained_model = train(model, trainloader, validloader, criterion, optimizer, device=device, epochs=args.epochs, print_every=args.print_every)

    test_model(model, testloader, device)
    save_model(model, args.checkpoint, 'vgg16', args.n_inputs, args.n_hidden, train_data.class_to_idx, optimizer)
    
# Run program
if __name__ == '__main__':
    main()
