import argparse
import os
import sys
import math

import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

# from models.model_retrieval import RetrievalModel
from models.model_retrieval_caption import RetrievalModel


import utils
from dataset import create_dataset, create_sampler, create_loader, build_tokenizer
from scheduler import create_scheduler
from optim import create_optimizer


def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itc_vs', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itc_vt', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itc_st', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100

    for i, (image, text_ls, captions, idx, cap_idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        text_inputs = []
        for text in text_ls:
            text_input = tokenizer(text, padding='longest', max_length=config['max_tokens'], return_tensors="pt").to(device)
            # longest
            # text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'], return_tensors="pt").to(device)
            # text_input = tokenizer(text, padding='longest', max_length=config['max_tokens'], return_tensors="pt").to(device)
            text_inputs.append(text_input)
        
        ###### generated caption
        captions = tokenizer(captions, padding='longest', max_length=config['max_tokens'], return_tensors="pt").to(device)
            
        loss_itc_vs, loss_itc_vt, loss_itc_st = model(image, text_inputs, captions=captions, idx=idx, epoch=epoch, cap_idx=cap_idx)
        loss = loss_itc_vs + loss_itc_vt + loss_itc_st
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        metric_logger.update(loss_itc_vs=loss_itc_vs.item())
        metric_logger.update(loss_itc_vt=loss_itc_vt.item())
        metric_logger.update(loss_itc_st=loss_itc_st .item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)

    # 得到关于文本的数据特征
    text_bs = config['batch_size_test_text']  # 256
    text_feats = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        # 分批次取文本数据
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)

        text_feat = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)  # last_hidden_state
        text_embed = model.get_features(text_embeds=text_feat)  # proj

        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)

    text_feats = torch.cat(text_feats, dim=0)  # torch.Size([1000, 40, 1024])
    text_embeds = torch.cat(text_embeds, dim=0)  # torch.Size([1000, 256])
    text_atts = torch.cat(text_atts, dim=0)

    # 得到关于图像的数据特征
    image_feats = []
    image_embeds = []
    for image, img_id in data_loader:
        image = image.to(device)

        image_feat, _ = model.get_vision_embeds(image)
        image_embed = model.get_features(image_embeds=image_feat)

        image_feats.append(image_feat)
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0)  # torch.Size([1000, 145, 1024])
    image_embeds = torch.cat(image_embeds, dim=0)  # torch.Size([1000, 256])

    # 1000 x 1000
    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(texts)), -100.0).to(device)

    num_tasks = utils.get_world_size()  # 1

    rank = utils.get_rank()  # 0

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    # sims_matrix 图像x文本
    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)  # k_test=128 找到和该图像最相似的128个文本

        # 把图像特征复制128次
        encoder_output = image_feats[start + i].repeat(config['k_test'], 1, 1)  # torch.Size([128, 145, 1024])
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        # 得到crossAtt的融合特征
        output = model.get_cross_embeds(image_embeds=encoder_output, image_atts=encoder_att,
                                        text_embeds=text_feats[topk_idx],
                                        text_atts=text_atts[topk_idx])  # torch.Size([128, 40, 1024])

        # 输入cls特征 model.itm_head(output[:, 0, :])--》torch.Size([128, 2]) matching概率
        score = model.itm_head(output[:, 0, :])[:, 1]  # torch.Size([128])

        score_matrix_i2t[start + i, topk_idx] = score

    # 转置 相似度变为 文本x图像
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step  # 0
    end = min(sims_matrix.size(0), start + step)  # 1000

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        # sim.shape torch.Size([1000]) 与1000个图像的相似度
        # 得到前128个最相似的图像
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx]  # torch.Size([128, 145, 1024])
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)  # torch.Size([128, 145])

        output = model.get_cross_embeds(image_embeds=encoder_output, image_atts=encoder_att,
                                        text_embeds=text_feats[start + i].repeat(config['k_test'], 1, 1),
                                        text_atts=text_atts[start + i].repeat(config['k_test'], 1))
        score = model.itm_head(output[:, 0, :])[:, 1]

        score_matrix_t2i[start + i, topk_idx] = score

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()




@torch.no_grad()
def evaluation_coarse_trans(model, data_loader, tokenizer, device, config, alpha=0.8):
    model.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text
    texts_trans = data_loader.dataset.text_trans

    num_text = len(texts)

    # 得到关于文本的数据特征
    text_bs = config['batch_size_test_text']  # 256
    text_feats = []
    text_embeds = []  
    text_atts = []

    text_feats_trans = []
    text_embeds_trans = []  
    text_atts_trans = []
    for i in range(0, num_text, text_bs):
        # 分批次取文本数据
        text = texts[i: min(num_text, i + text_bs)]
        text_trans = texts_trans[i: min(num_text, i + text_bs)]

        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)
        text_input_trans = tokenizer(text_trans, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)

        text_feat = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)  # last_hidden_state
        text_feat_trans = model.get_text_embeds(text_input_trans.input_ids, text_input_trans.attention_mask)  # last_hidden_state


        text_embed = model.get_features(text_embeds=text_feat, text_embeds_mask=text_input.attention_mask)  # proj
        text_embed_trans = model.get_features(text_embeds=text_feat_trans, text_embeds_mask=text_input_trans.attention_mask)  # proj

        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)

        text_embeds_trans.append(text_embed_trans)
        text_feats_trans.append(text_feat_trans)
        text_atts_trans.append(text_input_trans.attention_mask)


    text_feats = torch.cat(text_feats, dim=0)       # torch.Size([1000, 40, 1024])
    text_embeds = torch.cat(text_embeds, dim=0)     # torch.Size([1000, 256])
    text_atts = torch.cat(text_atts, dim=0)

    text_feats_trans = torch.cat(text_feats_trans, dim=0)       # torch.Size([1000, 40, 1024])
    text_embeds_trans = torch.cat(text_embeds_trans, dim=0)     # torch.Size([1000, 256])
    text_atts_trans = torch.cat(text_atts_trans, dim=0)

    # 得到关于图像的数据特征
    image_feats = []
    image_embeds = []
    for image, img_id in data_loader:
        image = image.to(device)

        image_feat, _ = model.get_vision_embeds(image)
        
        image_embed = model.get_features(image_embeds=image_feat)

        image_feats.append(image_feat)
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0)     # torch.Size([1000, 145, 1024])
    image_embeds = torch.cat(image_embeds, dim=0)   # torch.Size([1000, 256])

    # 1000 x 1000
    score_matrix_i2t = image_embeds @ text_embeds.t()
    score_matrix_i2t_trans = image_embeds @ text_embeds_trans.t()
    score_matrix_i2t = alpha * score_matrix_i2t + (1-alpha) * score_matrix_i2t_trans

    score_matrix_t2i = score_matrix_i2t.t()

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def evaluation_coarse(model, data_loader, tokenizer, device, config):
    model.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    texts = data_loader.dataset.text

    num_text = len(texts)

    # 得到关于文本的数据特征
    text_bs = config['batch_size_test_text']  # 256
    text_feats = []
    text_embeds = []  
    text_atts = []
    for i in range(0, num_text, text_bs):
        # 分批次取文本数据
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)

        text_feat = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)  # last_hidden_state

        text_embed = model.get_features(text_embeds=text_feat)  # proj

        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)

    text_feats = torch.cat(text_feats, dim=0)       # torch.Size([1000, 40, 1024])
    text_embeds = torch.cat(text_embeds, dim=0)     # torch.Size([1000, 256])
    text_atts = torch.cat(text_atts, dim=0)

    # 得到关于图像的数据特征
    image_feats = []
    image_embeds = []
    for image, generated_captions, img_id in data_loader:
        image = image.to(device)
        image_feat, _ = model.get_vision_embeds(image)
        
        ########  generated caption  ########
        caption_input = tokenizer(generated_captions, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)
        caption_embed = model.caption_proj(model.get_caption_embeds(caption_input.input_ids, caption_input.attention_mask))
        image_feat = model.crossattn(image_feat.transpose(0,1).contiguous(), caption_embed.transpose(0,1).contiguous())
        image_feat = image_feat.transpose(0,1).contiguous()
        ########  generated caption  ########
        
        image_embed = model.get_features(image_embeds=image_feat)

        image_feats.append(image_feat)
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0)     # torch.Size([1000, 145, 1024])
    image_embeds = torch.cat(image_embeds, dim=0)   # torch.Size([1000, 256])

    # 1000 x 1000
    score_matrix_i2t = image_embeds @ text_embeds.t()
    score_matrix_t2i = score_matrix_i2t.t()

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()



@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        # index表示第i个图像
        # 根据分数进行排名
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20

        # img2txt--dict; key--img_index, value--list--txt_index
        # 得到第i个图像对应的gt文本id集合
        # 这里应该是记录了集合里所有所文本最好的排名
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2
    txt_sumr = tr1 + tr5 + tr10
    img_sumr = ir1 + ir5 + ir10
    sumr_avg = np.round((txt_sumr + img_sumr) / 6, 2)

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'txt_sum_r': txt_sumr,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean,
                   'img_sumr': img_sumr,
                   'sumr_avg': sumr_avg,
                   'sumr_sum': (txt_sumr+img_sumr)}
    return eval_result



def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if args.epoch > 0:
        config['schedular']['epochs'] = args.epoch
        print(f"### set epochs to: {args.epoch}", flush=True)

    if args.bs > 0:
        config['batch_size_train'] = args.bs // world_size

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating retrieval dataset", flush=True)
    train_dataset, val_dataset_dict, test_dataset_dict = create_dataset('re', config)

    train_dataset_size = len(train_dataset)

    if utils.is_main_process():
        print(f"### Train Files: {[os.path.basename(rpath) for rpath in config['train_file']]}")
        print(f"### Train data {train_dataset_size}, batch size, {config['batch_size_train']}, world_size {world_size}")
        print(f"### Validation: {[(k, len(dataset)) for k, dataset in val_dataset_dict.items()]}")
        print(f"### Test: {[(k, len(dataset)) for k, dataset in test_dataset_dict.items()]}")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        train_sampler = create_sampler([train_dataset], [True], num_tasks, global_rank)
    else:
        train_sampler = [None]

    train_loader = create_loader([train_dataset], train_sampler, batch_size=[config['batch_size_train']],
                                 num_workers=[4],
                                 is_trains=[True],
                                 collate_fns=[None])[0]

    val_test_loader_set = {}
    for k in val_dataset_dict.keys():
        val_test_loader_set[k] = create_loader([val_dataset_dict[k], test_dataset_dict[k]], [None, None],
                                      batch_size=[config['batch_size_test']] * 2,
                                      num_workers=[4, 4], is_trains=[False, False], collate_fns=[None, None])

    print("Creating model", flush=True)
    model = RetrievalModel(config=config)
    model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate)
    
    ############################## 
    ##### 加载caption_encoder
    ##############################
    model.init_caption_encoder()
    
    
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    tokenizer = build_tokenizer(config['text_encoder'])

    print("### output_dir, ", args.output_dir, flush=True)

#     print("Running zero-shot evaluation", flush=True)
#     for language, [val_loader, test_loader] in val_test_loader_set.items():
#         score_test_i2t, score_test_t2i = evaluation_coarse(model_without_ddp, test_loader, tokenizer, device, config)

#         if utils.is_main_process():
#             test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img,
#                                    test_loader.dataset.img2txt)
#             print(f"{language}-test: {test_result}")
#         dist.barrier()
#     exit()

    print("Start training", flush=True)
    start_time = time.time()
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    arg_sche['step_per_epoch'] = math.ceil(train_dataset_size/(config['batch_size_train']*world_size))
    lr_scheduler = create_scheduler(arg_sche, optimizer)

    max_epoch = config['schedular']['epochs']
    best = 0
    best_epoch = 0
    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler, config)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        else:
            log_stats = {}

        r_mean = 0
        for language, [val_loader, test_loader] in val_test_loader_set.items():
            score_val_i2t, score_val_t2i, = evaluation_coarse(model_without_ddp, val_loader, tokenizer, device, config)
            score_test_i2t, score_test_t2i = evaluation_coarse(model_without_ddp, test_loader, tokenizer, device, config)

            if utils.is_main_process():
                val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)
                print(f"{language}-val: {val_result}")
                test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
                print(f"{language}-test: {test_result}")

                r_mean += test_result['r_mean']

                for k, v in val_result.items():
                    log_stats[f'{language}_val_{k}'] = v

                for k, v in test_result.items():
                    log_stats[f'{language}_test_{k}'] = v

            dist.barrier()

        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")

        if args.evaluate:
            break

        if utils.is_main_process():
            if r_mean > best:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'config': config,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                best = r_mean
                best_epoch = epoch

            elif epoch >= config['schedular']['epochs'] - 1:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                    # 'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    # 'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_{epoch}.pth'))

        dist.barrier()
        torch.cuda.empty_cache()

    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write("best epoch: %d" % best_epoch)

        os.system(f"cat {args.output_dir}/log.txt")

    dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)  # this script works for both mscoco and flickr30k
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')

    parser.add_argument('--epoch', default=-1, type=int)
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus")
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

    main(args, config)
