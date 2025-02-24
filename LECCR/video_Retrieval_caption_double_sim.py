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
from models.video_model_retrieval_caption import RetrievalModel

# +reg_loss+caption_loss

import utils
from dataset import create_dataset, create_sampler, create_loader, build_tokenizer, mbert_build_tokenizer
from scheduler import create_scheduler
from optim import create_optimizer
from clip import tokenize as clip_tokenizer

def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itc_vs', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itc_vt', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itc_st', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itc_c', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_reg_c', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100

    for i, (video, video_mask, text_ls, captions, idx, cap_idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        video = video.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        text_inputs = []
        for text in text_ls:
            text_input = tokenizer(text, padding='longest', max_length=config['max_tokens'], return_tensors="pt").to(device)
            # longest
            # text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'], return_tensors="pt").to(device)
            # text_input = tokenizer(text, padding='longest', max_length=config['max_tokens'], return_tensors="pt").to(device)
            text_inputs.append(text_input)
        
        ###### generated caption
        if config['generated_caption_type'] != 'feats':
            if config['caption_encoder_name'] == 'clip':
                captions = clip_tokenizer(captions).to(device)
            elif config['caption_encoder_name'] == 'mbert':
                captions = tokenizer(captions, padding='longest', max_length=config['max_tokens'], return_tensors="pt").to(device)
            else:
                print('error')
                exit()
            
        loss_itc_vs, loss_itc_vt, loss_itc_st, loss_itc_c, loss_reg_c = model(video, video_mask, text_inputs, captions=captions, idx=idx, epoch=epoch, cap_idx=cap_idx)
        loss = loss_itc_vs + loss_itc_vt + loss_itc_st + loss_itc_c + loss_reg_c
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        metric_logger.update(loss_itc_vs=loss_itc_vs.item())
        metric_logger.update(loss_itc_vt=loss_itc_vt.item())
        metric_logger.update(loss_itc_st=loss_itc_st.item())
        metric_logger.update(loss_itc_c=loss_itc_c.item())
        metric_logger.update(loss_reg_c=loss_reg_c.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}



def norm_score(t2v_all_errors):
    t2v_all_score = -t2v_all_errors
    t2v_all_score = t2v_all_score - torch.min(t2v_all_score)
    t2v_all_score = t2v_all_score / torch.max(t2v_all_score)
    return -t2v_all_score


@torch.no_grad()
def evaluation_coarse(model, data_loader, tokenizer, device, config, alpha=0.9):
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
    caption_embeds = []
    for video, mask_video, generated_captions, img_id in data_loader:
        video = video.to(device)
        mask_video = mask_video.to(device)
        image_feat, image_atts = model.get_vision_embeds(video, mask_video)
        
        ########  generated caption  ########
        if config['caption_encoder_name'] == 'clip':
            captions = clip_tokenizer(generated_captions).to(device)
            caption_embed = model.get_caption_embeds(captions)
        elif config['caption_encoder_name'] == 'mbert':
            caption_input = tokenizer(generated_captions, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)
            caption_embed = model.get_caption_embeds(caption_input.input_ids, caption_input.attention_mask)
        key_padding_mask = torch.zeros_like(captions).masked_fill_(captions == 0, 1).bool() \
            if config['caption_encoder_name'] == 'clip' else ~caption_input.attention_mask.bool() 
        
        image_feat, caption_embed, _ = model.interaction_with_caption(image_embeds=image_feat, caption_embeds=caption_embed, key_padding_mask=key_padding_mask, video_mask=image_atts)
        image_feat = image_feat.transpose(0,1).contiguous()
        ########  generated caption  ########
        
        image_embed = model.get_features(image_embeds=image_feat, vis_mask=mask_video.unsqueeze(-1))
        
        caption_embed = model.caption_proj1(caption_embed)

        # image_feats.append(image_feat)
        image_embeds.append(image_embed)
        caption_embeds.append(caption_embed)

    # image_feats = torch.cat(image_feats, dim=0)     # torch.Size([1000, 145, 1024])
    image_embeds = torch.cat(image_embeds, dim=0)   # torch.Size([1000, 256])
    caption_embeds = torch.cat(caption_embeds, dim=1)
    # 检测是否存在NaN
    has_nan = torch.isnan(caption_embeds).any().item()

    # 1000 x 1000
    score_matrix_i2t = image_embeds @ text_embeds.t()
    score_matrix_t2i = score_matrix_i2t.t()

    n, bsz, d = caption_embeds.shape
    c_sim = caption_embeds.reshape(-1,d) @ text_embeds.transpose(0,1)
    c_score_matrix_i2t = torch.max(c_sim.reshape(n, bsz, bsz), dim=0)[0] 
    c_score_matrix_t2i = c_score_matrix_i2t.t()

    score_matrix_i2t = alpha * norm_score(score_matrix_i2t) + (1.-alpha) * norm_score(c_score_matrix_i2t)
    score_matrix_t2i = alpha * norm_score(score_matrix_t2i) + (1.-alpha) * norm_score(c_score_matrix_t2i)

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t.contiguous(), op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i.contiguous(), op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()



@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):

        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20

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
    train_dataset, val_dataset_dict, test_dataset_dict = create_dataset('video', config)

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

    from dataset.retrieval_dataset_video import collate_fn
    train_loader = create_loader([train_dataset], train_sampler, batch_size=[config['batch_size_train']],
                                 num_workers=[4],
                                 is_trains=[True],
                                 collate_fns=[collate_fn])[0]

    from dataset.retrieval_dataset_video import collate_fn_eval
    val_test_loader_set = {}
    for k in val_dataset_dict.keys():
        val_test_loader_set[k] = create_loader([val_dataset_dict[k], test_dataset_dict[k]], [None, None],
                                      batch_size=[config['batch_size_test']] * 2,
                                      num_workers=[4, 4], is_trains=[False, False], collate_fns=[collate_fn_eval, collate_fn_eval])
    
    import time
    start_time = time.time()

    print("Creating model", flush=True)
    model = RetrievalModel(config=config)
    # model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate)
    
    ############################## 
    ##### 加载caption_encoder
    ##############################
    model.init_caption_encoder()
    
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.checkpoint != 'null':
        print('load checkpint ....')
        state_dict = torch.load(args.checkpoint, map_location=torch.device('cpu'))['model']
        mismatched_params = model.load_state_dict(state_dict, strict=False)
        print(mismatched_params)

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    tokenizer = mbert_build_tokenizer()

    print("### output_dir, ", args.output_dir, flush=True)

    # print("Running zero-shot evaluation", flush=True)
    # for language, [val_loader, test_loader] in val_test_loader_set.items():
    #     score_test_i2t, score_test_t2i = evaluation_coarse(model_without_ddp, test_loader, tokenizer, device, config)

    #     end_time = time.time()
    #     print('time is', end_time-start_time)
    #     if utils.is_main_process():
    #         test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img,
    #                                test_loader.dataset.img2txt)
    #         print(f"{language}-test: {test_result}")
    #     dist.barrier()
    # exit()

    


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
    start_epoch = 0

    #######################################
    # 加载断点
    #######################################
    RESUME = False
    if RESUME:
        path_checkpoint = "./model_parameter/test/ckpt_best_50.pth" 
        checkpoint = torch.load(path_checkpoint) 

        model.load_state_dict(checkpoint['model']) 
        optimizer.load_state_dict(checkpoint['optimizer']) 
        config = checkpoint['config'] 
        start_epoch = checkpoint['epoch'] 
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler']) 

    
    for epoch in range(start_epoch, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler, config)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        else:
            log_stats = {}

        sumr_sum = 0
        for language, [val_loader, test_loader] in val_test_loader_set.items():
            score_val_i2t, score_val_t2i, = evaluation_coarse(model_without_ddp, val_loader, tokenizer, device, config)
            score_test_i2t, score_test_t2i = evaluation_coarse(model_without_ddp, test_loader, tokenizer, device, config)

            if utils.is_main_process():
                val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)
                print(f"{language}-val: {val_result}")
                test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
                print(f"{language}-test: {test_result}")

                sumr_sum += test_result['sumr_sum']

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
            if sumr_sum > best:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                best = sumr_sum
                best_epoch = epoch

            elif epoch >= config['schedular']['epochs'] - 1:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_{epoch}.pth'))
            
            print(f'best epoch is {best_epoch} and best sumr is {best:.2f}')

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
