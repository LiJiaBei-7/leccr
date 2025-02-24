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


def getVideoId(cap_id):
    vid_id = cap_id.split('#')[0]
    if vid_id.endswith('.jpg') or vid_id.endswith('.mp4'):
        vid_id = vid_id[:-4]
    return vid_id



# using videochat2 generated captions
class re_train_dataset_caption(Dataset):
    def __init__(self, config, transform, max_words=30):

        self.transform = transform
        self.root_dir = config['root_dir'] 
        ann_files = config['train_file']
        self.image_root = config['image_root']
        self.dataset = config['dataset']
        self.caption_type = config['generated_caption_type']

        self.max_words = max_words
        self.img_ids = {}
        self.captions = {}
        self.cap_ids = []
        self.languages = []


        if self.dataset == 'mscoco':
            self.image_map = {}
            file = os.path.join(self.root_dir, 'img_id', 'image_ids.txt')
            with open(file) as f:
                for line in f.readlines():
                    tmp = line.strip('\n').split(' ', 1)
                    id, name = tmp[0], tmp[1]
                    self.image_map[id] = name


        
        # 读取videochat2生成的caption文件
        print('Loading caption...')
        s_time = time.time()
        self.generated_caption = config['generated_caption_dir']
        img_id_path = os.path.join(self.root_dir, 'img_id', 'train_id.txt')
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
        id_file = os.path.join(self.root_dir, 'img_id/train_id.txt')

        

        with open(id_file) as f:
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


        if self.dataset == 'mscoco':
            try:
                _image_id = self.image_map[image_id]
                image_path = f'{self.image_root}/{_image_id}'
            except:
                print(f'------{image_id}')
        else:
            image_path = f'{self.image_root}/{image_id}.jpg'

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        captions = []
        for k,v in self.captions.items():
            if k != 0:
                cap_id = cap_id.replace('#enc#', f'#enc2{self.languages[k-1]}#')
            caption = pre_caption(v[cap_id], self.max_words)
            captions.append(caption)
        return image, captions, generated_caption, self.img_ids[image_id], cap_id



    
class re_eval_dataset_caption(Dataset):
    def __init__(self, config, ann_file, transform, max_words=30, test_trans_file=None, type='eval'):
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

        if self.dataset == 'mscoco':
            self.image_map = {}
            id_trans_path = os.path.join(self.root_dir, 'img_id', 'image_ids.txt')
            with open(id_trans_path) as f:
                for line in f.readlines():
                    tmp = line.strip('\n').split(' ')
                    self.image_map[tmp[0]] = tmp[1]
            
            language = ann_file.split('.')[0].split('_')[-1]
            id_path = f'{language}_val_id.txt' if type == 'eval' else f'{language}_test_id.txt' 
 
        else:
            id_path = 'val_id.txt' if type == 'eval' else 'test_id_2016.txt'  

        self.generated_caption = config['generated_caption_dir']
        img_id_path = os.path.join(self.root_dir, 'img_id', id_path)
        self.generated_caption_dict = {}
        with open(img_id_path) as f:
            for line in f.readlines():
                line = line.strip('\n')
                if self.caption_type == 'feats':
                    file_path = os.path.join(self.generated_caption, f'{line}.npy')
                    self.generated_caption_dict[line] = torch.tensor(np.load(file_path, allow_pickle=True)).reshape(-1,768)
                else:
                    # image id
                    if self.dataset == 'mscoco':
                        file_path = os.path.join(self.generated_caption, f'{self.image_map[line].strip(".jpg")}.txt')
                    else:
                        file_path = os.path.join(self.generated_caption, f'{line}.txt')
                    with open(file_path, 'r') as file:
                        self.generated_caption_dict[line] = file.read().strip('\n')
                        # self.generated_caption_dict[line]= file.read().strip('\n').split('.')[0]+'.'
        
        # # 读取videochat2生成的caption文件
        # self.generated_caption = config['generated_caption_dir']
        # # 获取目录下所有文件的列表
        # file_list = os.listdir(self.generated_caption)
        # # 读取每个 txt 文件的内容
        # self.generated_caption_dict = {}
        # for txt_file in file_list:
        #     if self.caption_type == 'feats':
        #         key = txt_file.strip('.npy')
        #         file_path = os.path.join(self.generated_caption, txt_file)
        #         self.generated_caption_dict[key] = torch.tensor(np.load(file_path, allow_pickle=True))
        #     else:
        #         # image id
        #         key = txt_file.strip('.txt')
        #         file_path = os.path.join(self.generated_caption, txt_file)
        #         with open(file_path, 'r') as file:
        #             self.generated_caption_dict[key]= file.read().strip('\n').split('.')[0]+'.'


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


        # if self.dataset == 'mscoco':
        #     self.image_map = {}
        #     file = os.path.join(self.root_dir, 'image_ids.txt')
        #     with open(file) as f:
        #         for line in f.readlines():
        #             tmp = line.strip().split(' ', 1)
        #             id, name = tmp[0], tmp[1]
        #             self.image_map[id] = name

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_id = self.image[index]
        image_path = f'{self.image_root}/{image_id}'
        generated_caption = self.generated_caption_dict[image_id.split('.')[0]]

        if self.dataset == 'mscoco':
            try:
                _image_id = self.image_map[image_id.strip('.jpg')]
                image_path = f'{self.image_root}/{_image_id}'
            except:
                print('------', image_id)

        # image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, generated_caption, index




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

