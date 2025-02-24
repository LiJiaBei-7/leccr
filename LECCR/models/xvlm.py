
import os
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from functools import partial

from models.vit import interpolate_pos_embed
from models.clip_vit import CLIPVisionTransformer
from models.swin_transformer import  interpolate_relative_pos_embed

from utils import read_json


def load_params_change_prefix(state_dict: dict, prefix: str, new_prefix: str):
    if prefix == new_prefix:
        return state_dict

    state_dict_new = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            k = k.replace(prefix, new_prefix)

        state_dict_new[k] = v

    return state_dict_new



def load_params_choose_layers(prefix: str, state_dict: dict, mapper: dict):
    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            new_k = None
            for i in mapper.keys():
                if k.startswith(f'{prefix}.{i}.'):
                    new_k = k.replace(f'{prefix}.{i}.', f'{prefix}.{mapper[i]}.')
                    break

            if new_k:
                state_dict[new_k] = state_dict[k]

            del state_dict[k]

    return state_dict


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )


allgather = AllGather.apply


def build_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, input_dim * 2),
        nn.LayerNorm(input_dim * 2),
        nn.GELU(),
        nn.Linear(input_dim * 2, output_dim)
    )



def build_clip_encoder(config):
    import clip
    clip_name = 'ViT-B/32'
    vision_width = 512
    clip_encoder,_ = clip.load(clip_name, device='cuda')
    return clip_encoder.float(), vision_width


def build_text_encoder(config):
    from transformers import BertModel

    txt_bert_params = {
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
        }
    txt_bert_config = 'bert-base-multilingual-cased'

    text_encoder = BertModel.from_pretrained(txt_bert_config, output_hidden_states=False,
                                                return_dict=True, **txt_bert_params)
    text_width = 768
    return text_encoder, text_width


def load_pretrained(ckpt_rpath, config, is_eval=False, load_text=False):
    checkpoint = torch.load(ckpt_rpath, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint

    if is_eval:
        return state_dict

    num_patches = (config['image_res'] // config['patch_size']) ** 2

    print("### Loading pretrained vision encoder", flush=True)
    if config['use_clip_vit']:
        del state_dict['vision_encoder.position_ids']
        pos_embed_reshaped = interpolate_pos_embed(state_dict['vision_encoder.pos_embed.weight'].unsqueeze(dim=0),
                                                   num_patches=num_patches, num_extra_tokens=1)
        state_dict['vision_encoder.pos_embed.weight'] = pos_embed_reshaped.squeeze(dim=0)

    elif config['use_swin']:

        window_size = read_json(config['vision_config'])['window_size']

        for k in list(state_dict.keys()):
            if 'relative_position_bias_table' in k:
                dst_num_pos = (2 * window_size - 1) ** 2
                state_dict[k] = interpolate_relative_pos_embed(state_dict[k], dst_num_pos, param_name=k)
            elif ('relative_position_index' in k) or ('attn_mask' in k):
                del state_dict[k]

    else:
        pos_embed_reshaped = interpolate_pos_embed(state_dict['vision_encoder.pos_embed'],
                                                   num_patches=num_patches, num_extra_tokens=1)
        state_dict['vision_encoder.pos_embed'] = pos_embed_reshaped

    if load_text:
        print("### Loading pretrained text encoder", flush=True)
        for key in list(state_dict.keys()):
            if key.startswith('text_encoder.'):
                if 'bert.' in key:
                    encoder_key = key.replace('bert.', '')
                    state_dict[encoder_key] = state_dict[key]
                    del state_dict[key]

    return state_dict


class XVLMBase(nn.Module):
    def __init__(self, config=None, load_vision_params=False, load_text_params=False,
                 use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False,
                 config_text=None):
        super().__init__()
        self.allgather = allgather
        self.init_params = []  # train from scratch with larger lr

        self.clip_encoder, vision_width = build_clip_encoder(config=config) 

        self.text_encoder, text_width = build_text_encoder(config=config) 

        self.vision_width = vision_width
        self.text_width = text_width 

        if use_contrastive_loss:
            self.embed_dim = config['embed_dim']
            self.text_proj = nn.Linear(self.text_width, self.embed_dim)
            self.init_params.extend(['text_proj.' + n for n, _ in self.text_proj.named_parameters()])

            if config['use_one_cl_proj_only']:
                assert self.vision_width == self.text_width
                self.vision_proj = None
            else:
                self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
                self.init_params.extend(['vision_proj.' + n for n, _ in self.vision_proj.named_parameters()])

            self.temp = nn.Parameter(torch.ones([]) * config['temp'])
            self.init_params.extend(['temp'])

        if use_matching_loss:
            self.itm_head = build_mlp(input_dim=self.text_width, output_dim=2)
            self.init_params.extend(['itm_head.' + n for n, _ in self.itm_head.named_parameters()])

        if use_bbox_loss:
            self.bbox_head = build_mlp(input_dim=self.text_width, output_dim=4)
            self.init_params.extend(['bbox_head.' + n for n, _ in self.bbox_head.named_parameters()])

        # check
        named_parameters = set([n for n, _ in self.named_parameters()])
        for n in set(self.init_params):
            if n not in named_parameters:
                print(f"warning: {n} not in named_parameters")
                self.init_params.remove(n)

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)


    def get_vision_embeds(self, image, image_atts=None, idx_to_group_img=None, return_hidden=True):
        """
        vision_embeds: cls + patch embeds
        """
        if idx_to_group_img is None:
            _,image_embeds = self.clip_encoder.encode_image(image, return_hidden=return_hidden)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            return image_embeds, image_atts  # full attention

        else:  # image < bsz
            if image_atts is None:
                _,image_embeds_fullatts = self.clip_encoder.encode_image(image, return_hidden=return_hidden)

                image_embeds_fullatts = torch.gather(image_embeds_fullatts, dim=0,
                                                     index=idx_to_group_img.view(-1, 1, 1).expand(
                                                         -1, image_embeds_fullatts.shape[1],
                                                         image_embeds_fullatts.shape[2]))  # expend to bsz

                image_atts = torch.ones(image_embeds_fullatts.size()[:-1], dtype=torch.long).to(image.device)

                return image_embeds_fullatts, image_atts

            else:
                assert image_atts.size(0) == idx_to_group_img.size(0)  # bsz
                image_embeds, image_embeds_fullatts = \
                    self.clip_encoder.encode_image(image, return_hidden=return_hidden, idx_to_group_img=idx_to_group_img, image_atts=image_atts)

                image_embeds_fullatts = torch.gather(image_embeds_fullatts, dim=0,
                                                     index=idx_to_group_img.view(-1, 1, 1).expand(
                                                         -1, image_embeds_fullatts.shape[1],
                                                         image_embeds_fullatts.shape[2]))

                return image_embeds, image_atts, image_embeds_fullatts

    def get_text_embeds(self, text_ids, text_atts):
        return self.text_encoder(text_ids, attention_mask=text_atts, return_dict=True).last_hidden_state


    def get_features(self, image_embeds=None, text_embeds=None, vis_pooling='cls'):
        vision_proj = self.text_proj if self.vision_proj is None else self.vision_proj

        if image_embeds is None:
            return F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        elif text_embeds is None:
            if vis_pooling == 'cls':
                return F.normalize(vision_proj(image_embeds[:, 0, :]), dim=-1)
            elif vis_pooling == 'mean':
                return F.normalize(vision_proj(torch.mean(image_embeds, dim=1)), dim=-1)
            else:
                print('vis_pooling Error!')
                exit()
        else:
            return F.normalize(vision_proj(image_embeds[:, 0, :]), dim=-1), \
                   F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        
   

    def get_contrastive_loss(self, image_feat, text_feat, idx=None):
        """
        Args:
            image_feat, text_feat: normalized

        Returns: contrastive loss

        """
        assert image_feat.size(-1) == self.embed_dim
        assert text_feat.size(-1) == self.embed_dim

        image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        logits = image_feat_all @ text_feat_all.t() / self.temp

        bsz = image_feat_all.shape[0]

        if idx is None:
            labels = torch.arange(bsz, device=image_feat.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)

        else:
            idx = idx.view(-1, 1)
            assert idx.size(0) == image_feat.size(0)
            idx_all = allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
            pos_idx = torch.eq(idx_all, idx_all.t()).float()
            labels = pos_idx / pos_idx.sum(1, keepdim=True)

            loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()

        return (loss_i2t + loss_t2i) / 2


