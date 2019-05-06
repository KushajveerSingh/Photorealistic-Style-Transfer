# Photorealistic-Style-Transfer
PyTorch unofficial implementation of the paper [High Resolution Network for Photorealisitc Style Transfer](https://arxiv.org/pdf/1904.11617.pdf). Check out my blog post on this ["All you need for Photorealistic Style Transfer in PyTorch"](https://medium.com/@kushajreal/all-you-need-for-photorealistic-style-transfer-in-pytorch-acb099667fc8).

## Repository Structure
1. `src` -> Main code for the repo
    * [config.py](src/config.py)
    * [hrnet.py](src/hrnet.py) :- Implementation of all models
    * [style_transfer.py](src/style_transfer.py) :- Function to train model on your own images
    * [utils.py](src/utils.py) :- Function to load image, convert tensor to numpy image and compute gram matrix.
    * `imgs` :- Folder containing the images for training
2. `notebooks` -> The development notebooks I used when creating this repo. The .py scripts are a copy paste from these notebooks in most of the cases.
    * `create_models.ipynb` :- contains functions for `utils.py` and `hrnet.py`
    * `train_model.ipynb` :- Follow this notebook for training the model on you own images. It implements style_transfer.py code
    * `test_model.ipynb` :- Extra notebook that I used while testing my .py scripts and viewing the results using different values of content and style weights.

## How to use
```
usage: style_transfer.py [-h] [--img_root IMG_ROOT] [--c_img CONTENT_IMG]
                         [--s_img STYLE_IMG] [--save_dir SAVE_DIR]
                         [--use_gpu USE_GPU] [--c_size CONTENT_SIZE]
                         [--c_w CONTENT_WEIGHT] [--s_w STYLE_WEIGHT]

PyTorch implementation of Photorealistic Style transfer from the paperHigh-
Resolution Network for Photorealistic Style
Transfer(https://arxiv.org/pdf/1904.11617v1.pdf).

optional arguments:
  -h, --help            show this help message and exit
  --img_root IMG_ROOT   The root directory containing all your images. This is
                        where output will be stored.
  --c_img CONTENT_IMG, --content_img CONTENT_IMG
                        Path to the content image relative to img_root
  --s_img STYLE_IMG, --style_img STYLE_IMG
                        Path to the style image relative to img_root
  --save_dir SAVE_DIR   Directory location where to save style transfered
                        images
  --use_gpu USE_GPU     Bool: If true then use GPU else CPU
  --c_size CONTENT_SIZE, --content_size CONTENT_SIZE
                        Size of the content image to be used in model (must be
                        divisible by 4
  --c_w CONTENT_WEIGHT, --content_weight CONTENT_WEIGHT
                        Weight for content loss
  --s_w STYLE_WEIGHT, --style_weight STYLE_WEIGHT
                        Weight for style loss
```

Also, in `config.py` there are two dictionaries that you can modify to set your hyperparam values.
```python
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
```

The style_transfer.py file is mainly meant to be used in Jupyter Notebooks. But if you want to use it using the terminal, then change these lines accordingly.
```python
# If using jupyter notebook use plt.show() to show the plots.
# When using terminal comment this line as you would have to close
# your figures for training to proceed.
plt.show()

# Use this to save your figures to disk
plt.savefig(f'{args.save_dir}fig{i}.png')
i += 1
```

If you are confused about something refer to `test_model.ipynb` where I test with different images using the above code.

## Documentation
Most of the paper details and code description has been covered in the [blog post](https://medium.com/@kushajreal/all-you-need-for-photorealistic-style-transfer-in-pytorch-acb099667fc8).

## Dependencies
* PyTorch 1.0+
* tqdm
* PIL
* CUDA

`tqdm` is not mandatory. If you do not want to use it just modift this line in `src/style_transfer.py`

```python
for epoch in tqdm(range(cfg['steps']+1)):

# Modify to this
for epoch in range(cfg['steps']+1):
```

For CUDA and CUDNN refer to [pytorch](https://pytorch.org/) install guide.

## TODO
- [ ] Add video support
- [ ] Batch training
- [ ] Fastai conversion
- [ ] Summary on how to choose content and style weights
- [ ] Make an API for easy use (Not sure about this right now)
 
## License
[Apache License 2.0](LICENSE)