import torch
from models import XVLMBase, load_pretrained

import torch.nn.functional as F
import torch.nn as nn



class RetrievalModel(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=False, use_bbox_loss=False)

        self.init_params = []
        self.caption_encoder_name = config['caption_encoder_name']
        self.weight_caption_loss = config['weight_caption_loss']
        self.n_queries = config['num_queries']
        self.caption_ca_layer = config['caption_ca_layer']
        self.caption_interaction_layer = config['caption_interaction_layer']
        self.weight_reg_loss = config['weight_reg_loss']
        self.l2_criterion = nn.MSELoss()
        self.weight_cv_loss = config['weight_cv_loss']
        self.weight_dstl_loss = config['weight_dstl_loss']

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):

        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)
    
    def init_caption_encoder(self):
        if self.caption_encoder_name == 'mbert':
            self.caption_encoder = self.text_encoder
            self.caption_width = self.text_width
        elif self.caption_encoder_name == 'clip':
            self.caption_width = self.vision_width
        else:
            print('caption_width error')
            exit()

        self.caption_proj = nn.Linear(self.caption_width, self.vision_width)
        self.queries = nn.Parameter(torch.zeros(self.n_queries, 1, self.vision_width))

        from models.attention import CrossAttention, SelfAttention
        self.crossattn_query = CrossAttention(d_model=self.vision_width, nhead=8, num_layers=self.caption_ca_layer)
        self.crossattn = CrossAttention(d_model=self.vision_width, nhead=8, num_layers=self.caption_interaction_layer)
        self.crossattn2 = CrossAttention(d_model=self.vision_width, nhead=8, num_layers=self.caption_interaction_layer)
        self.caption_proj1 = nn.Linear(self.vision_width, self.embed_dim)

        self.cproj = nn.Linear(self.vision_width, self.vision_width)
        self.vproj = nn.Linear(self.vision_width, self.vision_width)
    
    def get_caption_embeds(self, text_ids, text_atts=None):
        
        if self.caption_encoder_name == 'clip':
            return self.clip_encoder.encode_text(text_ids, return_hidden=True)[-1]
        elif self.caption_encoder_name == 'mbert':
            return self.caption_encoder(text_ids, attention_mask=text_atts, return_dict=True).last_hidden_state
        else:
            print('caption_encoder_name error!')
            exit()

    def caption_regularization(self, caption_embeds):
        # caption_embeds n,bsz,d
        bsz = caption_embeds.size(1)
        caption_embeds = F.normalize(caption_embeds, p=2, dim=-1)
        diagonal_matrix = torch.eye(self.n_queries).unsqueeze(0).repeat(bsz, 1, 1).to(caption_embeds.device)
        caption_embeds = caption_embeds.transpose(0,1)
        sim = caption_embeds @ caption_embeds.transpose(1,-1) # bsz,n,n
        sim = sim - diagonal_matrix
        return sim.mean()

    
    def interaction_with_caption(self, image_embeds, caption_embeds, key_padding_mask):
        bsz = image_embeds.size(0)
        queries = self.queries.expand(-1, bsz, -1)
        caption_embeds = self.caption_proj(caption_embeds)
        caption_embeds = self.crossattn_query(queries, caption_embeds.transpose(0,1).contiguous(), memory_key_padding_mask=key_padding_mask)
        after_image_embeds = self.crossattn(image_embeds.transpose(0,1).contiguous(), caption_embeds, 
                                      memory_key_padding_mask=None)
        after_caption_embeds = self.crossattn2(caption_embeds, image_embeds.transpose(0,1).contiguous())
        return after_image_embeds, after_caption_embeds, caption_embeds


    def norm_score(self, score):
        score = score - torch.min(score)
        score = score / torch.max(score)
        return score
    
    
    # logits_sv + logits_sc
    def dstl_loss(self, image_embeds, caption_embeds, text_embeds_s, text_embeds_t, idx, alpha=0.8):
        image_embeds = self.allgather(image_embeds, torch.distributed.get_rank(), torch.distributed.get_world_size())
        caption_embeds = self.allgather(caption_embeds.transpose(0,1).contiguous(), torch.distributed.get_rank(), torch.distributed.get_world_size()).transpose(0,1).contiguous()
        text_embeds_s = self.allgather(text_embeds_s, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_embeds_t = self.allgather(text_embeds_t, torch.distributed.get_rank(), torch.distributed.get_world_size())

        logits_tv = text_embeds_t @ image_embeds.t()
        logits_sv = text_embeds_s @ image_embeds.t()

        n, bsz, d = caption_embeds.shape
        sim = caption_embeds.reshape(-1,d) @ text_embeds_s.transpose(0,1)
        logits_sc = torch.max(sim.reshape(n, bsz, bsz), dim=0)[0] 
        logits_sc = self.norm_score(logits_sc)
        logits_sv = self.norm_score(logits_sv)

        labels = alpha * logits_sv + (1.-alpha) * logits_sc
        labels = F.softmax(labels, 1)

        logits_tv = F.log_softmax(logits_tv, 1)

        loss = F.kl_div(logits_tv, labels.detach(), reduction='batchmean')

        return loss

    def caption_vision_loss(self, caption, image, idx):
        caption = caption.transpose(0,1).contiguous()
        image = self.allgather(image, torch.distributed.get_rank(), torch.distributed.get_world_size())
        caption = self.allgather(caption, torch.distributed.get_rank(), torch.distributed.get_world_size())

        caption = F.normalize(self.cproj(caption))
        image = F.normalize(self.vproj(image))

        bsz, vn, d = image.shape
        _, cn, _ = caption.shape

        _image = image.reshape(-1, d)
        _caption = caption.reshape(-1, d)

        sim = _caption @ _image.t()
        sim = sim.reshape(bsz, cn, bsz, vn).transpose(1,2)
        sim = torch.mean(torch.mean(sim, dim=-1), dim=-1)

        idx = idx.view(-1, 1)
        idx_all = self.allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
        pos_idx = torch.eq(idx_all, idx_all.t()).float()
        labels = pos_idx / pos_idx.sum(1, keepdim=True)

        loss = -torch.sum(F.log_softmax(sim, dim=1) * labels, dim=1).mean()

        return loss

    def get_caption_contrastive_loss(self, caption_embeds, text_feats):
        n, bsz, d = caption_embeds.shape
        sim = caption_embeds.reshape(-1,d) @ text_feats.transpose(0,1)
        logits = torch.max(sim.reshape(n, bsz, bsz), dim=0)[0] / self.temp
        labels = torch.arange(bsz, device=caption_embeds.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return (loss_t2i + loss_i2t) / 2

    def forward(self, image, text_ls, captions=None, idx=None, epoch=None, cap_idx=None):
        ori_image_embeds, image_atts = self.get_vision_embeds(image)
        
        ########  generated caption  ########
        with torch.no_grad():
            if self.caption_encoder_name != 'clip':
                caption_ids_s, caption_atts_s = captions.input_ids, captions.attention_mask
                caption_embeds = self.get_caption_embeds(caption_ids_s, caption_atts_s)
            else:
                caption_embeds = self.get_caption_embeds(captions, None)

        key_padding_mask = torch.zeros_like(captions).masked_fill_(captions == 0, 1).bool() \
            if self.caption_encoder_name == 'clip' else ~caption_atts_s.bool() 
        image_embeds, caption_embeds, ori_caption_embeds = self.interaction_with_caption(image_embeds=ori_image_embeds, 
                                                     caption_embeds=caption_embeds, key_padding_mask=key_padding_mask)
        image_embeds = image_embeds.transpose(0,1).contiguous() # bsz,n,d
        ########  generated caption  ########

        loss_cv = self.caption_vision_loss(ori_caption_embeds, ori_image_embeds, idx) * self.weight_cv_loss
        loss_reg_c = self.caption_regularization(ori_caption_embeds)
        
        image_feat = self.get_features(image_embeds, None)
        
        text_input_s = text_ls[0]
        text_ids_s, text_atts_s = text_input_s.input_ids, text_input_s.attention_mask
        text_embeds_s = self.get_text_embeds(text_ids_s, text_atts_s)
        text_feat_s = self.get_features(None, text_embeds_s)
        
        text_input_t = text_ls[1]
        text_ids_t, text_atts_t = text_input_t.input_ids, text_input_t.attention_mask
        text_embeds_t = self.get_text_embeds(text_ids_t, text_atts_t)
        text_feat_t = self.get_features(None, text_embeds_t)
        
        loss_itc_vs = self.get_contrastive_loss(image_feat, text_feat_s, idx=idx)
        loss_itc_vt = self.get_contrastive_loss(image_feat, text_feat_t, idx=idx)
        loss_itc_st = self.get_contrastive_loss(text_feat_s, text_feat_t, idx=idx)

        _caption_embeds = self.caption_proj1(caption_embeds)
        loss_itc_sc = self.get_caption_contrastive_loss(_caption_embeds, text_feat_s)
        loss_itc_tc = self.get_caption_contrastive_loss(_caption_embeds, text_feat_t)
        loss_itc_c = loss_itc_sc + loss_itc_tc

        loss_dstl = self.dstl_loss(image_feat, _caption_embeds, text_feat_s, text_feat_t, idx) * self.weight_dstl_loss
        loss_itc_vt  = loss_itc_vt * (1-self.weight_dstl_loss) + loss_dstl

        return loss_itc_vs+loss_cv, loss_itc_vt, loss_itc_st, loss_itc_c * self.weight_caption_loss, loss_reg_c * self.weight_reg_loss


