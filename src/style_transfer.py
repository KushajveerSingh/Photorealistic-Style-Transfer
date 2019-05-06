import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19 

import os
from PIL import Image
import numpy as np 
from tqdm import tqdm 
import matplotlib.pyplot as plt

from hrnet import HRNet
from utils import *
from config import *

def train_model(args, device, cfg=cfg):
    # Load VGG19 features
    vgg = vgg19(pretrained=True).features
    vgg = vgg.to(device)
    # We don't want to train VGG
    for param in vgg.parameters():
        param.requires_grad_(False)
        
    # Load style net
    style_net = HRNet()
    style_net = style_net.to(device)
    
    # Load images
    content_img = load_image(os.path.join(args.img_root, args.content_img), size=args.content_size)
    content_img = content_img.to(device)
    style_img = load_image(os.path.join(args.img_root, args.style_img))
    style_img = style_img.to(device)

    # Get features from VGG
    content_features = get_features(content_img, vgg)
    style_features = get_features(style_img, vgg)
    target = content_img.clone().requires_grad_(True).to(device)
    style_gram_matrixs = {layer: get_gram_matrix(style_features[layer]) for layer in style_features}

    optim = torch.optim.Adam(style_net.parameters(), lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=cfg['step_size'], gamma=cfg['gamma'])

    content_loss_epoch = []
    style_loss_epoch = []
    total_loss_epoch = []
    i = 0
    output_image = content_img
    content_weight = args.content_weight
    style_weight = args.style_weight

    # Start training
    for epoch in tqdm(range(cfg['steps']+1)):
        scheduler.step()

        target = style_net(content_img).to(device)
        target.requires_grad_(True)
        
        target_features = get_features(target, vgg)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        
        style_loss = 0
        
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram_matrix = get_gram_matrix(target_feature)
            style_gram_matrix = style_gram_matrixs[layer]
            
            layer_style_loss = style_weights[layer] * torch.mean((target_gram_matrix - style_gram_matrix) ** 2)
            b, c, h, w = target_feature.shape
            style_loss += layer_style_loss / (c*h*w)
            
        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss_epoch.append(total_loss.item())
        
        style_loss_epoch.append(style_weight * style_loss)
        content_loss_epoch.append(content_weight * content_loss.item())
        
        optim.zero_grad()
        total_loss.backward()
        optim.step()
        
        if epoch % cfg['show_every'] == 0:
            print("After %d criterions:" % epoch)
            print('Total loss: ', total_loss.item())
            print('Content loss: ', content_loss.item())
            print('Style loss: ', style_loss.item())
            plt.imshow(im_convert(target))
            plt.show()

            # plt.savefig(f'{args.save_dir}fig{i}.png')
            # i += 1

        output_image = target

    # return style_net

    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Load device and VGG
    if args.use_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            raise Exception('GPU is not available')
    else:
        device = torch.device('cpu')

    train_model(args, device)