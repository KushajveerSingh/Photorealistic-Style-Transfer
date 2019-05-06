[If you are seeing this then it means I am still updating my repo]

# Photorealistic-Style-Transfer
PyTorch unofficial implementation of the paper [High Resolution Network for Photorealisitc Style Transfer](https://arxiv.org/pdf/1904.11617.pdf). Check out my blog post on this ["All you need for Photorealistic Style Transfer in PyTorch"](https://medium.com/@kushajreal/all-you-need-for-photorealistic-style-transfer-in-pytorch-acb099667fc8).



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