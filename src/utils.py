import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

def load_image(path, size=None):
    """
    Resize img to size, size should be int and also normalize the
    image using imagenet_stats
    """
    img = Image.open(path)

    if size is not None:
        img = img.resize((size, size))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    img = transform(img).unsqueeze(0)
    return img


def im_convert(img):
    """
    Convert img from pytorch tensor to numpy array, so we can plot it.
    It follows the standard method of denormalizing the img and clipping
    the outputs
    
    Input:
        img :- (batch, channel, height, width)
    Output:
        img :- (height, width, channel)
    """
    img = img.to('cpu').clone().detach()
    img = img.numpy().squeeze(0)
    img = img.transpose(1, 2, 0)
    img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    img = img.clip(0, 1)
    return img


def get_features(img, model, layers=None):
    """
    Use VGG19 to extract features from the intermediate layers.
    """
    if layers is None:
        layers = {
            '0': 'conv1_1',   # style layer
            '5': 'conv2_1',   # style layer
            '10': 'conv3_1',  # style layer
            '19': 'conv4_1',  # style layer
            '28': 'conv5_1',  # style layer
            
            '21': 'conv4_2'   # content layer
        }
    
    features = {}
    x = img
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features


def get_gram_matrix(img):
    """
    Compute the gram matrix by converting to 2D tensor and doing dot product
    img: (batch, channel, height, width)
    """
    b, c, h, w = img.size()
    img = img.view(b*c, h*w)
    gram = torch.mm(img, img.t())
    return gram