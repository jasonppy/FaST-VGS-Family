Transformer-based visually grounded speech models:
FaST-VGS:
![fast-vgs](./pics/archi1.png "FaST-VGS")
FaST-VGS+:
![fast-vgs-plus](./pics/archi2.png "FaST-VGS+")

Code for papers:

```
@inproceedings{peng2022fastvgs,
  title={Fast-Slow Transformer for Visually Grounding Speech},
  author={Peng, Puyuan and Harwath, David},
  booktitle={Proceedings of the 2022 International Conference on Acoustics, Speech and Signal Processing},
  year={2022}
}

@inproceedings{peng2022fastvgsplus,
  title={Self-Supervised Representation Learning for Speech Using Visual Grounding and Masked Language Modeling},
  author={Peng, Puyuan and Harwath, David},
  booktitle={The Self-Supervised Learning for Speech and Audio Processing Workshop at AAAI 2022},
  year={2022}
}
```
## 0. Clone repo and install requirements


## 1. Model weights
Will be up for downloading this week!

## 2. Use the trained model
```python
import sys
import os
import pickle
sys.path.append("the path to FaST-VGS-Family")
model_path = "path to weights and args you download in 1."
from models import fast_vgs, w2v2_model
# load args
with open(f"{model_path}/args.pkl", "rb") as f:
    args = pickle.load(f)
# load weights
weights = torch.load(os.path,join(model_path, "best_bundle.pth"))

# if want to use the entire model for e.g. speech-image retrieval
dual_encoder = fast_vgs.DualEncoder(args)
cross_encoder = fast_vgs.CrossEncoder(args)
dual_encoder.load_state_dict(weights['dual_encoder'])
cross_encoder.load_state_dict(weights['cross_encoder'])

# if only want to use the audio branch for e.g. feature extraction for speech downstream tasks
w2v2_model.Wav2Vec2Model_cls(args)
w2v2_model.carefully_load_state_dict(weights['dual_encoder']) # will filter out weights that don't belong to w2v2

```

*** For speech-image retrieval or training models from scratch, please follow the steps below ***
## 3. Download data

### Image
Note that for Places and flickr8k, we download the raw images and extract Faster RCNN regional features in the next step, for MSCOCO, we directly download the Faster RCNN feature distributed by [Hao Tan](https://www.cs.unc.edu/~airsplay/) in [LXMERT](https://github.com/airsplay/lxmert).

```bash
data_root="root for store data"

# COCO
# images train2014 (17 GB) and val2014 (8 GB)
coco_root=${data_root}/coco
mkdir ${data_root}/coco
wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip -P ${coco_root}/mscoco_imgfeat
unzip ${coco_root}/mscoco_imgfeat/train2014_obj36.zip -d ${coco_root}/mscoco_imgfeat && rm ${coco_root}/mscoco_imgfeat/train2014_obj36.zip
wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip -P ${coco_root}/mscoco_imgfeat
unzip ${coco_root}/mscoco_imgfeat/val2014_obj36.zip -d -d ${coco_root}/mscoco_imgfeat && rm ${coco_root}/mscoco_imgfeat/val2014_obj36.zip
# spoken captions (64G)
wget https://data.csail.mit.edu/placesaudio/SpokenCOCO.tar.gz -P ${coco_root}
cd ${coco_root}
tar -xf SpokenCOCO.tar.gz


# Places 
# Images
# follow http://places.csail.mit.edu/downloadData.html

# spoken captions (85G)
places_root=${data_root}/places
wget https://data.csail.mit.edu/placesaudio/placesaudio_2020_splits.tar.gz -P ${places_root}
cd ${places_root}
tar -xf placesaudio_2020_splits.tar.gz


# Flickr8k
flickr8k_root=${data_root}/flickr8k
# images
# download e.g. from https://www.kaggle.com/adityajn105/flickr8k/activity

# spoken captions 
wget https://groups.csail.mit.edu/sls/downloads/flickraudio/downloads/flickr_audio.tar.gz -P ${flickr8k_root} 
cd ${flickr8k_root}
tar -xf flickr_audio.tar.gz
```

## 4. Image feature preprocessing
We first extract Faster RCNN features using the Docker image released by Hao Tan. The instructions are at https://github.com/airsplay/lxmert#faster-r-cnn-feature-extraction, see section "Yet Another Example: Feature Extraction for MS COCO Images" for how the MSCOCO features are extracted. I put feature extraction scripts for Places and Flickr8k in `./datasets/preprocessing/extract_faster_rcnn` in this repo. Please mount this folder in Docker so it's easier for your to do Faster RCNN feature extraction.

After the the features are extracted, we generate hdf5 files and some helper files for using dataset scripts in `./datasets`. before that, we unroll the json file of coco. simply put the coco_root in `./datasets/preprocessing/unroll_coco.py`

After this, we can generate hdf5 and other files directly used by the dataset scripts. Change the roots in `./datasets/generate_hdf5_coco_places_flickr8k_imgfeat.py` and run this file.

## 5. Training scripts
The scripts in in `./scripts`


### Acknowledgement
Model code uses the [wav2vec2 code](https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/README.md) from [fairseq](https://github.com/pytorch/fairseq) and [LXMERT code](https://github.com/airsplay/lxmert) from [Hao Tan](https://www.cs.unc.edu/~airsplay/).