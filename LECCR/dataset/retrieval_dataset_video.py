import json
import os

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile

import numpy as np
import torch

import time
import re

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption


def collate_fn(batch):
    # frames_tensor, captions, generated_caption, self.img_ids[image_id], cap_id
    frames, captions_en, captions_zh, generated_caption, img_ids, cap_ids = zip(*batch)
    
    # 创建填充后的输入序列张量和掩码张量
    input_lengths = [len(seq) for seq in frames]
    max_input_length = max(input_lengths)
    frame_vec_len = len(frames[0][0])

    padded_frames = torch.zeros(len(frames), max_input_length, frame_vec_len)
    mask_frames = torch.zeros(len(frames), max_input_length).bool()

    for i, seq in enumerate(frames):
        end = input_lengths[i]
        padded_frames[i, :end] = torch.tensor(seq[:end])
        mask_frames[i, :end] = 1
    return padded_frames, mask_frames, [captions_en, captions_zh], list(generated_caption), torch.tensor(img_ids), list(cap_ids)


def collate_fn_eval(batch):
    # frames_tensor, captions, generated_caption, self.img_ids[image_id], cap_id
    frames, generated_caption, index = zip(*batch)

    # 创建填充后的输入序列张量和掩码张量
    input_lengths = [len(seq) for seq in frames]
    max_input_length = max(input_lengths)
    frame_vec_len = len(frames[0][0])

    padded_frames = torch.zeros(len(frames), max_input_length, frame_vec_len)
    mask_frames = torch.zeros(len(frames), max_input_length).bool()

    for i, seq in enumerate(frames):
        end = input_lengths[i]
        padded_frames[i, :end] = torch.tensor(seq[:end])
        mask_frames[i, :end] = 1

    return padded_frames, mask_frames, list(generated_caption), torch.tensor(index)



def getVideoId(cap_id):
    vid_id = cap_id.split('#')[0]
    if vid_id.endswith('.jpg') or vid_id.endswith('.mp4'):
        vid_id = vid_id[:-4]
    return vid_id



from utils.bigfile import BigFile
# using videochat2 generated captions
class video_train_dataset_caption(Dataset):
    def __init__(self, config, transform, visual_feats, video2frames, max_words=30):

        self.transform = transform
        self.root_dir = config['root_dir'] 
        ann_files = config['train_file']
        self.visual_feature = config['image_root']
        self.dataset = config['dataset']
        self.caption_type = config['generated_caption_type']

        self.max_words = max_words
        self.img_ids = {}
        self.captions = {}
        self.cap_ids = []
        self.languages = []

        self.visual_feats =visual_feats
        visual_feat_dim = visual_feats.ndims    # msrvtt 4096
        self.video2frames = video2frames

        # 读取videochat2生成的caption文件
        print('Loading caption...')
        s_time = time.time()
        self.generated_caption = config['generated_caption_dir']
        img_id_path = os.path.join(self.root_dir, 'video_id', 'train_id.txt')
        self.generated_caption_dict = {}
        with open(img_id_path) as f:
            for line in f.readlines():
                line = line.strip('\n')
                if self.caption_type == 'feats':
                    file_path = os.path.join(self.generated_caption, f'{line}.npy')
                    self.generated_caption_dict[line] = torch.tensor(np.load(file_path, allow_pickle=True)).reshape(-1,768)
                else:
                    if self.dataset == 'mscoco':
                        file_path = os.path.join(self.generated_caption, f'{self.image_map[line].strip(".jpg")}.txt')
                    else:
                        # image id
                        file_path = os.path.join(self.generated_caption, f'{line}.txt')
                    with open(file_path, 'r') as file:
                        self.generated_caption_dict[line] = file.read().strip('\n')
        e_time = time.time()
        print(f"Execution Time - Load caption: {e_time-s_time:.2f} seconds")

        for i, ann_file in enumerate(ann_files):
            if i != 0:
                language = ann_file.rsplit('/', 1)[-1].split('.', 1)[0].split('2', 1)[-1]
                self.languages.append(language)

            caption_map = {}
            ann_file = os.path.join(self.root_dir, ann_file)
            with open(ann_file, 'r') as cap_reader:
                for line in cap_reader.readlines():
                    cap_id, caption = line.strip().split(' ', 1)
                    caption_map[cap_id] = caption
                    if i == 0:
                        self.cap_ids.append(cap_id)
            self.captions[i] = caption_map

        self.length = len(self.cap_ids)
    
        with open(img_id_path) as f:
            for i, line in enumerate(f.readlines()):
                id = line.strip('\n')
                self.img_ids[id] = i


    def __len__(self):
        return len(self.captions[0])

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        image_id = getVideoId(cap_id)
        ######## generated captions
        generated_caption = self.generated_caption_dict[image_id]

        # video
        frame_list = self.video2frames[image_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feats.read_one(frame_id))
        frames_tensor = torch.Tensor(frame_vecs)

        captions = []
        for k,v in self.captions.items():
            if k != 0:
                cap_id = cap_id.replace('#enc#', f'#enc2{self.languages[k-1]}#')
            caption = pre_caption(v[cap_id], self.max_words)
            captions.append(caption)
        return frames_tensor, captions[0], captions[1], generated_caption, self.img_ids[image_id], cap_id



    
class video_eval_dataset_caption(Dataset):
    def __init__(self, config, ann_file, transform, visual_feats, video2frames, max_words=30, test_trans_file=None, type='eval'):
        self.transform = transform
        self.image_root = config['image_root']
        self.dataset = config['dataset']
        self.root_dir = config['root_dir']
        self.max_words = max_words
        self.caption_type = config['generated_caption_type']

        self.text = []
        self.cap_ids = []
        self.txt2img = {}
        self.img2txt = {}
        image_ids = {}
        self.image = []

        self.visual_feats =visual_feats
        visual_feat_dim = visual_feats.ndims
        self.video2frames = video2frames

        id_path = 'val_id.txt' if type == 'eval' else 'test_id.txt'  

        self.generated_caption = config['generated_caption_dir']
        img_id_path = os.path.join(self.root_dir, 'video_id', id_path)
        self.generated_caption_dict = {}
        with open(img_id_path) as f:
            for line in f.readlines():
                line = line.strip('\n')
                if self.caption_type == 'feats':
                    file_path = os.path.join(self.generated_caption, f'{line}.npy')
                    self.generated_caption_dict[line] = torch.tensor(np.load(file_path, allow_pickle=True)).reshape(-1,768)
                else:
                    # image id
                    file_path = os.path.join(self.generated_caption, f'{line}.txt')
                    with open(file_path, 'r') as file:
                        self.generated_caption_dict[line] = file.read().strip('\n')
        
        n = 0
        ann_file = os.path.join(self.root_dir, ann_file)
        with open(ann_file, 'r') as cap_reader:
            for txt_id, line in enumerate(cap_reader.readlines()):
                cap_id, caption = line.strip().split(' ', 1)

                image_id = getVideoId(cap_id)
                if image_id in image_ids.keys():
                    img_id = image_ids[image_id]
                else:
                    img_id = n
                    image_ids[image_id] = img_id
                    self.image.append(image_id)
                    n += 1
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt.setdefault(img_id, [])
                self.txt2img[txt_id] = img_id
                self.img2txt[img_id].append(txt_id)
                self.cap_ids.append(cap_id)

        if test_trans_file is not None:
            self.text_trans = []
            with open(test_trans_file, 'r') as cap_reader:
                for txt_id, line in enumerate(cap_reader.readlines()):
                    cap_id, caption = line.strip().split(' ', 1)
                    self.text_trans.append(pre_caption(caption, self.max_words))


    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        
        # video
        image_id = self.image[index]
        frame_list = self.video2frames[image_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feats.read_one(frame_id))
        frames_tensor = torch.Tensor(frame_vecs)

        generated_caption = self.generated_caption_dict[image_id.split('.')[0]]

        return frames_tensor, generated_caption, index




class re_train_dataset(Dataset):
    def __init__(self, config, transform, max_words=30):

        self.transform = transform
        self.root_dir = config['root_dir'] 
        ann_files = config['train_file']
        self.image_root = config['image_root']
        self.dataset = config['dataset']

        self.max_words = max_words
        self.img_ids = {}
        self.captions = {}
        self.cap_ids = []
        self.languages = []


        for i, ann_file in enumerate(ann_files):
            if i != 0:
                language = ann_file.rsplit('/', 1)[-1].split('.', 1)[0].split('2', 1)[-1]
                self.languages.append(language)

            caption_map = {}
            ann_file = os.path.join(self.root_dir, ann_file)
            with open(ann_file, 'r') as cap_reader:
                for line in cap_reader.readlines():
                    cap_id, caption = line.strip().split(' ', 1)
                    caption_map[cap_id] = caption
                    if i == 0:
                        self.cap_ids.append(cap_id)
            self.captions[i] = caption_map

        self.length = len(self.cap_ids)
        id_file = os.path.join(self.root_dir, 'FeatureData/train_id.txt')

        if self.dataset == 'mscoco':
            self.image_map = {}
            file = os.path.join(self.root_dir, 'image_ids.txt')
            with open(file) as f:
                for line in f.readlines():
                    tmp = line.strip().split(' ', 1)
                    id, name = tmp[0], tmp[1]
                    self.image_map[id] = name

        with open(id_file) as f:
            for i, line in enumerate(f.readlines()):
                id = line.strip('\n')
                self.img_ids[id] = i


    def __len__(self):
        return len(self.captions[0])

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        image_id = getVideoId(cap_id)
        image_path = f'{self.image_root}/{image_id}.jpg'

        if self.dataset == 'mscoco':
            try:
                _image_id = self.image_map[image_id]
                image_path = f'{self.img_path}/{_image_id}'
            except:
                print('------', image_id)

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        captions = []
        for k,v in self.captions.items():
            if k != 0:
                cap_id = cap_id.replace('#enc#', f'#enc2{self.languages[k-1]}#')
            caption = pre_caption(v[cap_id], self.max_words)
            captions.append(caption)
        return image, captions, self.img_ids[image_id], cap_id





class re_eval_dataset(Dataset):
    def __init__(self, config, ann_file, transform, max_words=30, test_trans_file=None, type='eval'):
        self.transform = transform
        self.image_root = config['image_root']
        self.dataset = config['dataset']
        self.root_dir = config['root_dir']
        self.max_words = max_words

        self.text = []
        self.cap_ids = []
        self.txt2img = {}
        self.img2txt = {}
        image_ids = {}
        self.image = []

        n = 0
        ann_file = os.path.join(self.root_dir, ann_file)
        with open(ann_file, 'r') as cap_reader:
            for txt_id, line in enumerate(cap_reader.readlines()):
                cap_id, caption = line.strip().split(' ', 1)

                image_id = getVideoId(cap_id)
                if image_id in image_ids.keys():
                    img_id = image_ids[image_id]
                else:
                    img_id = n
                    image_ids[image_id] = img_id
                    self.image.append(image_id + '.jpg')
                    n += 1
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt.setdefault(img_id, [])
                self.txt2img[txt_id] = img_id
                self.img2txt[img_id].append(txt_id)
                self.cap_ids.append(cap_id)

        if test_trans_file is not None:
            self.text_trans = []
            with open(test_trans_file, 'r') as cap_reader:
                for txt_id, line in enumerate(cap_reader.readlines()):
                    cap_id, caption = line.strip().split(' ', 1)
                    self.text_trans.append(pre_caption(caption, self.max_words))


        if self.dataset == 'mscoco':
            self.image_map = {}
            file = os.path.join(self.root_dir, 'image_ids.txt')
            with open(file) as f:
                for line in f.readlines():
                    tmp = line.strip().split(' ', 1)
                    id, name = tmp[0], tmp[1]
                    self.image_map[id] = name

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_id = self.image[index]
        image_path = f'{self.image_root}/{image_id}'

        if self.dataset == 'mscoco':
            try:
                _image_id = self.image_map[image_id.strip('.jpg')]
                image_path = f'{self.image_root}/{_image_id}'
            except:
                print('------', image_id)

        # image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index

