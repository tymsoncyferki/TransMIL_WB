# TransMIL-visualisation

# configuration

put .h5 files from CLAM's extract_features_fp.py in the h5-files folder  
put WSI thumbnails in the images folder
put model.ckpt to model folder

Modify the source code of nystrom_attention

# running

run main.py

parameters:

downsample

The maximum size of the WSI is 10000x10000, the size of the thumbnail is 100x100, then the downsample is 10000/100=100.

patch_size

your patch_size at 20x magnification is 224x224, so here it should be 448x448
