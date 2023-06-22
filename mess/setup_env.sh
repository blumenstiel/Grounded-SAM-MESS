# run script with
# bash mess/setup_env.sh

# Create new environment "groundedsam"
conda create --name groundedsam -y python=3.8
source ~/miniconda3/etc/profile.d/conda.sh
conda activate groundedsam

# Install requirements for Grounded-SAM
conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
pip install segment-anything-py
pip install groundingdino-py
pip install huggingface_hub

# Install packages for dataset preparation
pip install gdown
pip install kaggle
pip install rasterio
pip install pandas