# Imports

import argparse
import json
import PIL
import torch
import numpy as np
from PIL import Image
from math import ceil
from train import check_gpu
from torchvision import models
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


def argparse_predict():
    '''
    This function is to define the argument parser for training
    '''
    # Parser definition
    parser = argparse.ArgumentParser(description='Neural network training parameters')
    
    parser.add_argument('--checkpoint',
                        type=str,
                        help='check point file name (string)'
                       )
    
    parser.add_argument('--image',
                        type=str,
                        help='image to be predicted with location - filepath (str)'
                       )
    
    parser.add_argument('--top_k',
                        type=int,
                        help='Top k matches in prediction (integer)'
                       )
    
    parser.add_argument('--category_names',
                        type=str,
                        help='categories to name mapping (string)'
                       )
    
    # GPU option
    parser.add_argument('--gpu',
                        action='store_true',
                        help='Use GPU for training or not'
                       )
    
    return parser.parse_args()

def load_model(filepath):
    '''
    Function that loads a checkpoint and rebuilds the model.
    '''
    checkpoint = torch.load(filepath)
    
    # download the same base architecture and freeze the parameters
    
    if checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif checkpoint['arch'] == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif checkpoint['arch'] == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif checkpoint['arch'] == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print('Model architecture provided is not suppored')
        raise 
        
    # Freezing the parameters    
    for param in model.parameters():
        param.requires_grad = False
        
    # set the classifier to match
    classifier = nn.Sequential(nn.Linear(checkpoint['inputs'], checkpoint['hidden']),
                           nn.ReLU(),
                           nn.Dropout(p=0.5),
                           nn.Linear(checkpoint['hidden'], len(checkpoint['class_to_idx'])),
                           nn.LogSoftmax(dim=1))
    
    model.classifier = classifier
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    
    return model

def process_image(image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img_loader = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()
    
    np_image = np.array(pil_image)    
            
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, cat_to_name, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    # setting model to evaluate()
    model.eval()
    image = process_image(image_path)
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    image_tensor.resize_([1, 3, 224, 224])
    model.to('cpu')
    result = torch.exp(model(image_tensor))
    ps, index = result.topk(topk)
    ps, index = ps.detach(), index.detach()
    ps.resize_([topk])
    index.resize_([topk])
    ps, index = ps.tolist(), index.tolist()
    label_index = []
    class_labels = {val: key for key, val in model.class_to_idx.items()}
    for i in index:
        label_index.append(int(class_labels[int(i)]))
    labels = []
    for i in label_index:
        labels.append(cat_to_name[str(i)])
    return ps, labels, label_index

def main():
    
    # Get the args for training
    args = argparse_predict()
    
    # Load the class/category names
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    # Load the model
    
    model = load_model(args.checkpoint)
    
    # Process image
    #img_tensor = process_image()
    
    # check gpu
    device = check_gpu(gpu_arg = args.gpu)
    print('Using', device)
    
    probs, labels, label_indices = predict(args.image, model, cat_to_name, args.top_k)
    
    print(probs)
    print(labels)
    print(label_indices)
    
# Run program
if __name__ == '__main__':
    main()
