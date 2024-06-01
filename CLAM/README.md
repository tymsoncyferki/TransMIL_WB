How to set up on windows (original readme in docs directory):

delete openslide from env file

`conda env create -f env.yml` <br>
`conda activate clam_latest`

download binary openslide and put ddl file in conda env (https://openslide.org/download/) <br>
`pip install openslide-python`

change sthresh in biopsy to 3

`python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 32 --preset bwh_biopsy.csv --seg --patch` <br>
(optionaly change patch size to 16 / 32 / 64 etc. and DATA/RESULTS_DIRECTORY name)

`python extract_features_fp.py --data_h5_dir RESULTS_DIRECTORY --data_slide_dir DATA_DIRECTORY --csv_path RESULTS_DIRECTORY/process_list_autogen.csv --feat_dir FEATURES_DIRECTORY --batch_size 64 --slide_ext .tif` <br>
(change DATA/RESULTS/FEATURES_DIRECTORY name)
