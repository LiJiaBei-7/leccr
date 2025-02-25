import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision.transforms import InterpolationMode

def build_tokenizer(text_encoder: str):
    tokenizer = XLMRobertaTokenizer.from_pretrained(text_encoder)
    tokenizer.add_special_tokens({'bos_token': tokenizer.cls_token, 'eos_token': tokenizer.sep_token})
    return tokenizer

def mbert_build_tokenizer():
    from transformers import BertTokenizer
    mpath = 'bert-base-multilingual-cased'
    # cache_dir = 'data/cache_dir'
    tokenizer = BertTokenizer.from_pretrained(mpath, dd_special_tokens=True)
    torch.cuda.empty_cache()
    return tokenizer


from dataset.pretrain_dataset_multilingual import ImageMultiTextDataset, RegionMultiTextDataset, ImageMonoTextDataset, ParaTextDataset

from dataset.retrieval_dataset import re_train_dataset, re_eval_dataset, re_train_dataset_caption, re_eval_dataset_caption

from dataset.retrieval_dataset_video import video_train_dataset_caption, video_eval_dataset_caption

from dataset.randaugment import RandomAugment

from transformers import XLMRobertaTokenizer


def read_dict(filepath):
    f = open(filepath,'r')  
    a = f.read()  
    dict_data = eval(a)  
    f.close()
    return dict_data

def create_dataset(dataset, config):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    pretrain_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.2, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        # RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              # 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    train_transform_wohflip = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        # transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    box_transform = transforms.Compose([
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'pretrain_multilingual':
        if len(config['train_file']):
            general_dataset = ImageMultiTextDataset(config, config['train_file'], rank=int(os.environ.get('RANK') or 0),
                                                    world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True,
                                                    repeat=True, transform=pretrain_transform)
        else:
            general_dataset = None

        if len(config['train_file_regions']):
            region_dataset = RegionMultiTextDataset(config, config['train_file_regions'],
                                                    rank=int(os.environ.get('RANK') or 0),
                                                    world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True,
                                                    repeat=True, transform=pretrain_transform, box_transform=box_transform)
        else:
            region_dataset = None

        if len(config['train_file_mono']):
            print("### not debugged yet")
            mono_dataset = ImageMonoTextDataset(config, config['train_file_mono'], rank=int(os.environ.get('RANK') or 0),
                                                world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True,
                                                repeat=True, transform=pretrain_transform)
        else:
            mono_dataset = None

        if len(config['train_file_text']):
            text_dataset = ParaTextDataset(config, config['train_file_text'], rank=int(os.environ.get('RANK') or 0),
                                           world_size=int(os.environ.get('WORLD_SIZE') or 1), shuffle=True, repeat=True)
        else:
            text_dataset = None

        return general_dataset, region_dataset, mono_dataset, text_dataset

    elif dataset == 're_caption':
        train_dataset = re_train_dataset_caption(config, train_transform)

        val_dataset_dict = {}
        for k, rpath in config['val_file'].items():
            val_dataset_dict[k] = re_eval_dataset_caption(config, rpath, test_transform)

        test_dataset_dict = {}
        for k, rpath in config['test_file'].items():
            test_dataset_dict[k] = re_eval_dataset_caption(config, rpath, test_transform, type='test', test_trans_file=config['test_trans_file'])

        return train_dataset, val_dataset_dict, test_dataset_dict
    
    elif dataset == 're':
        train_dataset = re_train_dataset(config, train_transform)

        val_dataset_dict = {}
        for k, rpath in config['val_file'].items():
            val_dataset_dict[k] = re_eval_dataset(config, rpath, test_transform)

        test_dataset_dict = {}
        for k, rpath in config['test_file'].items():
            test_dataset_dict[k] = re_eval_dataset(config, rpath, test_transform, type='test', test_trans_file=config['test_trans_file'])

        return train_dataset, val_dataset_dict, test_dataset_dict

    elif dataset == 'video':
        from utils.bigfile import BigFile
        visual_feat_path = os.path.join(config['root_dir'], 'FeatureData', config['image_root'])
        visual_feats = BigFile(visual_feat_path)
        visual_feat_dim = visual_feats.ndims
        video2frames = (os.path.join(config['root_dir'], 'FeatureData', config['image_root'], 'video2frames.txt'))
        video2frames = read_dict(video2frames)

        train_dataset = video_train_dataset_caption(config, train_transform, visual_feats, video2frames)

        val_dataset_dict = {}
        for k, rpath in config['val_file'].items():
            val_dataset_dict[k] = video_eval_dataset_caption(config, rpath, test_transform, visual_feats, video2frames)

        test_dataset_dict = {}
        for k, rpath in config['test_file'].items():
            test_dataset_dict[k] = video_eval_dataset_caption(config, rpath, test_transform, visual_feats, video2frames, type='test', test_trans_file=config['test_trans_file'])

        del visual_feats, video2frames
        return train_dataset, val_dataset_dict, test_dataset_dict
    
    else:
        raise NotImplementedError(f"dataset == {dataset}")


def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list, dim=0), question_list, answer_list, torch.Tensor(weight_list), n


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

    if len(loaders) <= 1:
        print(f"### be careful: func create_loader returns a list length of {len(loaders)}")

    return loaders
