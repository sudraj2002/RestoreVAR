# [ICLR-2026] RestoreVAR

[Sudarshan Rajagopalan](https://sudraj2002.github.io/) | [Kartik Narayan](https://kartik-3004.github.io/portfolio/) | [Vishal M. Patel](https://scholar.google.com/citations?user=AkEXTbIAAAAJ&hl=en)

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://sudraj2002.github.io/restorevarpage/) [![PDF](https://img.shields.io/badge/OpenReview-Forum-blue)](https://openreview.net/pdf?id=yvXtCn2zfz) [![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2505.18047) 

Code for RestoreVAR: Visual Autoregressive Generation for All-in-One Image Restoration.

## Getting Started

### Environment

Create the environment as follows

```commandline
conda create -n var_test python=3.9 -y
conda activate var_test
pip install -r requirements.txt
```

### Downloads

Download checkpoints [file](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/sambasa2_jh_edu/IQBRgls-pEXOSpunUEA4j0MtAeJzsjjMRHYRUowCnKwOMKU?e=EadbiN) 
and extract it to ```ckpts/```.

Download [training](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/sambasa2_jh_edu/IQAlFCaxzzntRYINFEZlyOhJAXsd2z4S2chSrRSl2OMXDQg?e=xnCGEL) and 
[testing](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/sambasa2_jh_edu/IQCetMYCEA6OQrPxKO7LJhDEAZ7op_CHiJZV4_CmervGIUU?e=yFeFm5)
data and extract them to ```data/``` and ```test_data/```, respectively.

Download meta-info ```json``` files for [testing](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/sambasa2_jh_edu/IQAmqEbEiK9RQoAprWt3K9-zASnAxy-xNDwD7t1ULXjKAG8?e=8guj8g)
and extract them to ```test_jsons/```.

## Testing

To get results of Table 1, run 

```commandline
bash test_regular.sh
```

To get results of Table 2, run

```commandline
bash test_generalization.sh
bash metric_generalization.sh
```

The scripts can be modified as needed.

## Training

For training the VAR transformer for restoration, run

```commandline
bash train.sh
```

The script loads the pretrained ```ckpts/var_d16.pth``` model and introduces the proposed components to train the 
model for restoration.

Once trained, the latent refiner transformer (LRT) can be trained using 

```commandline
bash train_refiner.sh
```

Prior to running this command, rename the existing ```local_output/``` directory to something else. The refiner uses
the ```ckpts/vae_restorevar.ckpt``` file which contains weights for the VAE decoder fine-tuned to handle continuous
latents.

The above commands use ```trainset.json``` and ```valset.json``` which contain information about file paths, 
datasets, etc. To include your own datasets, make ```.json``` files with entries as follows:

```commandline
{
    "image_path": <degraded_image_path>, 
    "target_path": <target_image_path>, 
    "degradation": <degradation>, 
    "dataset": <dataset>
}
```

## Citation

If you found our work useful, please cite:

```commandline
@inproceedings{
    rajagopalan2026restorevar,
    title={Restore{VAR}: Visual Autoregressive Generation for All-in-One Image Restoration},
    author={Sudarshan Rajagopalan and Kartik Narayan and Vishal M. Patel},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=yvXtCn2zfz}
}
```

## Acknowledgments

Our code uses parts from [VAR](https://github.com/FoundationVision/VAR) and [VARSR](https://github.com/quyp2000/VARSR). We thank the authors for sharing their codes!
