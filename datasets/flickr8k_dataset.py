import json
import numpy as np
import os
import torch
import torch.nn.functional
import soundfile as sf
from torch.utils.data import Dataset
import h5py
import pickle
import logging
logger = logging.getLogger(__name__)


class ImageCaptionDataset(Dataset):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--data_root", type=str, default="/data1/scratch/datasets_pyp/flickr8k")
        parser.add_argument("--raw_audio_base_path", type=str, default="/home/harwath/data/flickr8k_spoken_captions")
        parser.add_argument("--img_feat_len", type=int, help="num of img feats we will use", choices=list(range(1,37)), default=36)
        parser.add_argument("--audio_feat_len", type=float, help="maximal audio length", default=8)
        parser.add_argument("--val_audio_feat_len", type=float, help="maximal audio length", default=10.)

    def __init__(self, args, split = "train"):
        self.args = args
        self.split = split
        self.audio_feat_len = args.audio_feat_len if "train" in split else args.val_audio_feat_len
        split_name = split.replace("val", "dev") if not self.args.test else "test" # just in case you input val, instead of dev
        audio_dataset_json_file = os.path.join(args.data_root, f"flickr8k_{split_name}.json")
        img_dataset_h5py_file = os.path.join(args.data_root, f"img_feat/flickr8k_{split_name}_obj36.hdf5")
        img_id2index_file = os.path.join(args.data_root, f"flickr8k_{split_name}_imgid2idex.json")
        imgid2ordered_indices_file = os.path.join(args.data_root, f"flickr8k_{split_name}_imgid2ordered_indices.pkl")
        self.audio_base_path = args.raw_audio_base_path
        with open(audio_dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.data = data_json['data']
        self.img_data = h5py.File(img_dataset_h5py_file, 'r')
        with open(img_id2index_file, 'r') as fp:
            self.img_id2index = json.load(fp)
        with open(imgid2ordered_indices_file, 'rb') as f:
            self.img_id2ordered_indices = pickle.load(f)

    def _LoadAudio(self, path):
        x, sr = sf.read(path, dtype = 'float32')
        assert sr == 16000
        length_orig = len(x)
        if length_orig > 16000 * self.audio_feat_len:
            audio_length = int(16000 * self.audio_feat_len)
            x = x[:audio_length]
            x_norm = (x - np.mean(x)) / np.std(x)
            x = torch.FloatTensor(x_norm) 
        else:
            audio_length = length_orig
            new_x = torch.zeros(int(16000 * self.audio_feat_len))
            x_norm = (x - np.mean(x)) / np.std(x)
            new_x[:audio_length] = torch.FloatTensor(x_norm) 
            x = new_x        
        return x, audio_length

    def _LoadImage(self, index, ordered_indices):
        img_h, img_w = self.img_data['img_h'][index],self.img_data['img_w'][index]
        boxes = self.img_data['boxes'][index][ordered_indices[:self.args.img_feat_len]]
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        return torch.from_numpy(self.img_data['features'][index][ordered_indices[:self.args.img_feat_len]]), torch.from_numpy(boxes)
 
    def __getitem__(self, index):
        datum = self.data[index]
        img_id = datum['image']
        img_index = self.img_id2index[img_id]
        ordered_indices = self.img_id2ordered_indices[img_id]
        wavpath = os.path.join(self.audio_base_path, "wavs", datum['wav'])
        audio, nframes = self._LoadAudio(wavpath)
        mask_feats, boxes = self._LoadImage(img_index, ordered_indices)
        return mask_feats, boxes, audio, nframes, img_id, datum['wav']

    def __len__(self):
        return len(self.data)

    def collate(self, batch):
        vals = list(zip(*batch))
        collated = {}
        collated['visual_feats'] = torch.stack(vals[0])
        collated['boxes'] = torch.stack(vals[1])
        collated['audio'] = torch.nn.utils.rnn.pad_sequence(vals[2], batch_first=True)
        collated['audio_length'] = torch.LongTensor(vals[3])
        collated['img_id'] = np.array(vals[4])
        collated['fn'] = vals[5]
        collated['audio_attention_mask'] = torch.arange(len(collated['audio'][0])).unsqueeze(0) >= collated['audio_length'].unsqueeze(1)
        return collated

