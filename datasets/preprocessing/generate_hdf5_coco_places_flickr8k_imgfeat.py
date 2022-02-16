import base64
import numpy as np
import json
# import cv2
import csv
from tqdm import tqdm

import h5py
from pathlib import Path
import os
import sys
import csv
import base64
import time
import argparse
import numpy as np

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.
    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):
            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            # objects_conf is the prediction confidence ([0,1]) for corresponding objects_id
            # they are ordered descendingly according to confidence (so feats are also order this way, cause they have to match)
            # therefore, attrs_id are not ordered by attrs_conf but objects_conf
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)
            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data



def generate_h5py(audio_dataset_json_file, img_dataset_hdf5, img_id2index_file, img_data, places = False, flickr8k=False):
    with open(audio_dataset_json_file, 'r') as fp:
        data_json = json.load(fp)
    audio_data = data_json['data']

    all_img_id = set()
    if places or flickr8k:
        for item in audio_data:
            all_img_id.add(item['image'])
    else:
        for item in audio_data:
            all_img_id.add(item['image'].split("/")[-1].split(".")[0])
    n = len(all_img_id)
    print(f"total img used: {n}")

    img_id2index = {}
    print(f"dump image data into {img_dataset_hdf5}")

    Path(os.path.dirname(img_dataset_hdf5)).mkdir(parents=True, exist_ok=True)
    f = h5py.File(img_dataset_hdf5, 'w')
    utf8 = h5py.string_dtype('utf-8', 27)
    dset_img_id = f.create_dataset('img_id', (n,), dtype=utf8) # COCO_val2014_000000325114 or l/laundromat/gsun_df5b011043802b4049b632a005de20e3.jpg for Places
    dset_img_h = f.create_dataset('img_h', shape = (n,), dtype='int64')
    dset_img_w = f.create_dataset('img_w', shape = (n,), dtype='int64')
    dset_features = f.create_dataset('features', (n, 36, 2048), dtype='float32')
    dset_boxes = f.create_dataset('boxes', shape = (n, 36, 4), dtype='float32')
    dset_objects_id = f.create_dataset('objects_id', shape = (n, 36), dtype='int64')
    dset_objects_conf = f.create_dataset('objects_conf', shape = (n, 36), dtype='float32')
    dset_attrs_id = f.create_dataset('attrs_id', shape = (n, 36), dtype='int64')
    dset_attrs_conf = f.create_dataset('attrs_conf', shape = (n, 36), dtype='float32')
    
    start = time.time()
    
    for index, img_id in enumerate(all_img_id):
        # img_id = audio_data[i]['image'].split("/")[-1].split(".")[0]
        # if img_id not in img_id2index:
        img_id2index[img_id] = index
        datum = img_data[img_id]
        # datum = img_data[list(img_data.keys())[i]]
        dset_img_id[index] = np.array(img_id.encode('utf-8'), dtype=utf8)
        dset_img_h[index] = np.array(int(datum['img_h']))
        dset_img_w[index] = np.array(int(datum['img_w']))
        dset_features[index,:,:] = datum['features']
        dset_boxes[index,:,:] = datum['boxes']
        dset_objects_id[index,:] = datum['objects_id']
        dset_objects_conf[index,:] = datum['objects_conf']
        dset_attrs_id[index,:] = datum['attrs_id']
        dset_attrs_conf[index,:] = datum['attrs_conf']


        if index % 5000 == 0:
            t = time.time() - start
            print('processed %d / %d images (%.fs)' % (index, n, t))
            # break
    with open(img_id2index_file, "w") as f:
        json.dump(img_id2index, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['coco', 'places', 'flickr8k'])
    args = parser.parse_args()
    if args.dataset == 'coco':
        train_audio_dataset_json_file="/data1/scratch/coco_pyp/SpokenCOCO/SpokenCOCO_train_unrolled.json"
        val_audio_dataset_json_file = "/data1/scratch/coco_pyp/SpokenCOCO/SpokenCOCO_val_unrolled.json"
        train_img_dataset_tsv_file = "/data1/scratch/coco_pyp/coco_img_feat/train2014_obj36.tsv"
        val_img_dataset_tsv_file ="/data1/scratch/coco_pyp/coco_img_feat/val2014_obj36.tsv"

        train_img_dataset_hdf5 = "/data1/scratch/coco_pyp/coco_img_feat/SpokenCOCO_train_imgfeat.hdf5"
        val_img_dataset_hdf5 = "/data1/scratch/coco_pyp/coco_img_feat/SpokenCOCO_val_imgfeat.hdf5"

        train_imgid2index_file = "/data1/scratch/coco_pyp/SpokenCOCO/SpokenCOCO_train_imgid2idex.json"
        val_imgid2index_file = "/data1/scratch/coco_pyp/SpokenCOCO/SpokenCOCO_val_imgid2idex.json"

        img_data_train = load_obj_tsv(train_img_dataset_tsv_file)
        img_data_val = load_obj_tsv(val_img_dataset_tsv_file)

        img_data = {}
        for img_datum in img_data_train:
            img_data[img_datum['img_id']] = img_datum
        for img_datum in img_data_val:
            img_data[img_datum['img_id']] = img_datum

        generate_h5py(train_audio_dataset_json_file, train_img_dataset_hdf5, train_imgid2index_file, img_data)
        generate_h5py(val_audio_dataset_json_file, val_img_dataset_hdf5, val_imgid2index_file, img_data)
    elif args.dataset == 'places':
        json_root = "/data1/scratch/places_hdf5_pyp/metadata"
        tsv_root = "/data1/scratch/places_hdf5_pyp/places_imgfeat"
        hdf5_root = "/data1/scratch/places_hdf5_pyp/places_imgfeat"
        # splits = ['train_2020', 'dev_seen_2020', 'dev_unseen_2020', 'test_seen_2020', 'test_unseen_2020']
        splits = ['train_2020']
        for split in splits:
        # train_audio_dataset_json_file="/data1/scratch/coco_pyp/SpokenCOCO/SpokenCOCO_train_unrolled.json"
            audio_dataset_json_file = os.path.join(json_root, split+".json")
            img_dataset_tsv_file = os.path.join(tsv_root, split+"_obj36.tsv")
            img_dataset_hdf5 = os.path.join(hdf5_root, split+".hdf5")
            
            imgid2index_file =os.path.join(json_root, split+"_imgid2idex.json") 
            
            img_data_temp = load_obj_tsv(img_dataset_tsv_file)
            img_data = {}
            for img_datum in img_data_temp:
                # print(img_datum['img_id'])
                # raise
                img_data[img_datum['img_id']] = img_datum

            generate_h5py(audio_dataset_json_file, img_dataset_hdf5, imgid2index_file, img_data, places=True)

    elif args.dataset == 'flickr8k':
        train_audio_dataset_json_file="/home/harwath/data/flickr8k_spoken_captions/flickr8k_train.json"
        val_audio_dataset_json_file = "/home/harwath/data/flickr8k_spoken_captions/flickr8k_dev.json"
        test_audio_dataset_json_file = "/home/harwath/data/flickr8k_spoken_captions/flickr8k_test.json"

        train_img_dataset_tsv_file = "/data1/scratch/datasets_pyp/flickr8k/img_feat/flickr8k_train_obj36.tsv"
        val_img_dataset_tsv_file = "/data1/scratch/datasets_pyp/flickr8k/img_feat/flickr8k_dev_obj36.tsv"
        test_img_dataset_tsv_file = "/data1/scratch/datasets_pyp/flickr8k/img_feat/flickr8k_test_obj36.tsv"

        train_img_dataset_hdf5 = "/data1/scratch/datasets_pyp/flickr8k/img_feat/flickr8k_train_obj36.hdf5"
        val_img_dataset_hdf5 = "/data1/scratch/datasets_pyp/flickr8k/img_feat/flickr8k_dev_obj36.hdf5"
        test_img_dataset_hdf5 = "/data1/scratch/datasets_pyp/flickr8k/img_feat/flickr8k_test_obj36.hdf5"

        train_imgid2index_file = "/data1/scratch/datasets_pyp/flickr8k/flickr8k_train_imgid2idex.json"
        val_imgid2index_file = "/data1/scratch/datasets_pyp/flickr8k/flickr8k_dev_imgid2idex.json"
        test_imgid2index_file = "/data1/scratch/datasets_pyp/flickr8k/flickr8k_test_imgid2idex.json"
        
        img_data_train = load_obj_tsv(train_img_dataset_tsv_file)
        img_data_val = load_obj_tsv(val_img_dataset_tsv_file)
        img_data_test = load_obj_tsv(test_img_dataset_tsv_file)
        
        print("validation")
        img_data = {}
        for img_datum in img_data_val:
            img_data[img_datum['img_id']] = img_datum
        generate_h5py(val_audio_dataset_json_file, val_img_dataset_hdf5, val_imgid2index_file, img_data, flickr8k=True)

        print("test")
        img_data = {}
        for img_datum in img_data_test:
            img_data[img_datum['img_id']] = img_datum
        generate_h5py(test_audio_dataset_json_file, test_img_dataset_hdf5, test_imgid2index_file, img_data, flickr8k=True)

        print("train")
        img_data = {}
        for img_datum in img_data_train:
            img_data[img_datum['img_id']] = img_datum
        generate_h5py(train_audio_dataset_json_file, train_img_dataset_hdf5, train_imgid2index_file, img_data, flickr8k=True)