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
## 1. Model weights
Will be up for download this week!

## 2. Use the trained model
```python
import sys
import os
import pickle
sys.path.append("the path to FaST-VGS-Family")
model_path = "path to weights and args you download in 1."
from models import fast_vgs, w2v2_model
with open(f"{model_path}/args.pkl", "rb") as f:
    args = pickle.load(f)
# if want to use the entire model for e.g. speech-image retrieval
dual_encoder = fast_vgs.DualEncoder(args)
cross_encoder = fast_vgs.CrossEncoder(args)
weights = torch.load(os.path,join(model_path, "best_bundle.pth"))
dual_encoder.load_state_dict(weights['dual_encoder'])
cross_encoder.load_state_dict(weights['cross_encoder'])

# if only want the audio branch for e.g. feature extraction for speech downstream tasks
w2v2_model.Wav2Vec2Model_cls(args)
w2v2_model.carefully_load_state_dict(weights['dual_encoder']) # will filter out weights that don't belong to w2v2

```

If you want to train the models
## 3. Download data
### Image

### Audio

## 4. Image feature preprocessing

## 5. Training scripts