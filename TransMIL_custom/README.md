How to set up on windows:

conda create -n transmil python=3.7 -y
conda activate transmil

## conda install
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

# install related package
pip install -r requirements.txt

modify ``` q *= scale ``` to ``` q = q * self.scale ``` in nystrom_attention.py (in env packages)

to train on cpu:
python train.py --stage='train' --config='Bisque/TransMIL.yaml' --fold=0

to train on gpu:
python train.py --stage='train' --config='Bisque/TransMIL.yaml'  --gpus=1 --fold=0
(or gpus=0 idk)

to test:
python train.py --stage='test' --config='Bisque/TransMIL.yaml' --fold=0