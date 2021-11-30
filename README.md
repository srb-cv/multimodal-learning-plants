# multimodal-learning-plants
train_snp_image.py can be used to train the image and SNP modalities together. 

To see the documentation use command:\
python train_snp_image.py --help


This repository can be used to train three modalities---image, snp, and weather---and their sub-modalities e.g. different wavelengths of an image or different choromosomes in a SNP array. We create a model for each sub-modality and fuse the output from each model to obtain the trait value. All sub-modalites are trained jointly in an end to end manner. The first step is the data-preprocessing and then the training scripts can be used as explained below. There are three main traininig scripts that can be used to train the three modalities separately or in a combined fashion. 

**Data pre-processing**\
The JSON file should be preprocessed using the script _data_preprocessing/mapping_data.py_ to convert the file into a desirable CSV format.

**1. Train snp data**: The script _train_snp.py_ can be used to train on all or specific chromosomes of the snp modality. The following script can be used to train on all or specific chromosomes data for a specific trait. The importance of each sub-modality is computed automatically amd logged on Tensorboard.
```
python train_snp.py --dataset-csv "/path_to_preprcessed_csv" --batch-size 32 --gpus 1 --max_epochs 100 --min_epochs 20 --latent-dim 8 --bins A1 A2
```
_--bins denotes the chromosomes you want to train upon, where each chromosome act as a sub-modality_\
_--latent-dim denotes the number of output neurons in the model trained for every sub-modality. The output dimension of after fusing all sub-modalities is always 1 for regression_ \
Note: for extended documentation run _python train_snp.py --help_


**2. Train image data** the script _train_flowering.py_ can be used to train on all or specific wavelengths of the image modality. The following sample script can be used to train on specific wavelengths.
```
python train_flowering.py --dataset-csv /path_to_csv --data-root /path_to_image_folders --batch-size 128 --latent-dim 32 --wave-lens 0nm 530nm --max_epochs 100 --val-split 0.3
```
_--wave-lens denotes the wavelengths you want to train upon. 0nm correponds to the RGB image._
Note: for detailed documentation run the script _python train_flowering.py --help_

**3. Train combined image and snp data**: The script _train_snp_image.py_ can be used to train on the image and the snp modalities together. The follwoing sample script can be used to train on specific sub-modalities of images and snp data.
```
python train_snp_image.py --dataset-csv /path_to_csv --data-root /path_to_imag_folders --gpu 1 --num-workers 8 --batch-size 32 --latent-dim-image 32 --latent-dim-snp 4 --wave-lens 0nm --bins A7 A3 C3 C7 A4 A9 --max_epochs 100 --year-split MAL_2019
```
