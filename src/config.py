import argparse

def get_parser():
    parser = argparse.ArgumentParser(description=(
        "PyTorch implementation of Photorealistic Style transfer from the paper"
        "High-Resolution Network for Photorealistic Style Transfer(https://arxiv.org/pdf/1904.11617v1.pdf)."

        # "By default to work with this project you have to place your images in src/imgs folder with"
        # "names as content1.png, style1.png and so on. There should be equal number of content"
        # "and style images. This is done so that we can do style transfer on a batch."

        # "You also can pass the images with arguments."
        ))

    # Path of images
    parser.add_argument('--img_root', type=str, default='imgs/', dest='img_root',
        help="The root directory containing all your images. This is where output will be stored.")
    parser.add_argument('--c_img', '--content_img', type=str, default='content1.png', dest='content_img',
        help="Path to the content image relative to img_root")
    parser.add_argument('--s_img', '--style_img', type=str, default='style1.png', dest='style_img',
        help="Path to the style image relative to img_root")
    parser.add_argument('--save_dir', type=str, default='imgs/', dest='save_dir',
        help='Directory location where to save style transfered images')
    
    # Batch training
    # parser.add_argument('--use_batch', type=bool, default=False, dest='use_batch',
    #     help=("Bool, if you want to use batchs for style transfer. There are two modes for using this"
    #           "You can have one style image for every content image or you can have one style image for complete batch"))
    # parser.add_argument('--b', '--batch_size', type=int, default=16, dest='bs',
    #     help="The batch size to use in case use_batch=True")
    
    # GPU usage
    parser.add_argument('--use_gpu', type=bool, default=True, dest='use_gpu',
        help='Bool: If true then use GPU else CPU')
        
    parser.add_argument('--c_size', '--content_size', type=int, default=500, dest='content_size',
        help='Size of the content image to be used in model (must be divisible by 4')
    
    # Content and style weights
    parser.add_argument('--c_w', '--content_weight', type=int, default=150, dest='content_weight',
        help='Weight for content loss')
    parser.add_argument('--s_w', '--style_weight', type=int, default=1, dest='style_weight',
        help='Weight for style loss')
    


    return parser
    
    
style_weights = {
    'conv1_1': 0.1,
    'conv2_1': 0.2,
    'conv3_1': 0.4,
    'conv4_1': 0.8,
    'conv5_1': 1.6,
}

# For constant learning rate
cfg = {
    'lr': 5e-3,
    'show_every': 100,
    'steps': 1000,
    'step_size': 200,
    'gamma': 0.9
}