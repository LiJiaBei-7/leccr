import torch
from models import XVLMBase, load_pretrained

import torch.nn.functional as F


class RetrievalModel(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=False, use_bbox_loss=False)

        self.num_attention_heads = self.text_encoder.config.num_attention_heads
        self.init_params = []

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):

        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)
    
#     def get_imcst_loss(self, a, b):
#         a_all = self.allgather(a, torch.distributed.get_rank(), torch.distributed.get_world_size())
#         b_all = self.allgather(b, torch.distributed.get_rank(), torch.distributed.get_world_size())

#         # 计算两个张量之间的mse距离
#         mse_matrix = torch.mean(torch.pow(a_all.unsqueeze(1) - b_all.unsqueeze(0), 2), dim=2)
#         contrastive_labels = torch.arange(mse_matrix.size(0), device=mse_matrix.device)
#         aux_contrastive_loss = F.cross_entropy(mse_matrix, contrastive_labels)
        
#         return aux_contrastive_loss

    
    def forward(self, image, text_ls, idx=None, epoch=None, cap_idx=None):
        # loss_itc = torch.tensor(0.).cuda()
        image_embeds, image_atts = self.get_vision_embeds(image)
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
        loss_itc_st = self.get_contrastive_loss(text_feat_s, text_feat_t, idx=None)
        
        
        # loss_itc += self.get_imcst_loss(image_feat, text_feat) * 0.5

            # if i == 0:
            #     text_embeds_s = self.get_text_embeds(text_ids, text_atts)
            #     text_feat_s = self.get_features(None, text_embeds_s, text_atts)
            #     loss_itc += self.get_contrastive_loss(image_feat, text_feat_s, idx=idx)
            # else:
            #     text_embeds = self.get_text_embeds(text_ids, text_atts)

            #     text_feat = self.get_features(None, text_embeds, text_atts)

            #     # cross-modal
            #     loss_itc += self.get_contrastive_loss(image_feat, text_feat, idx=idx)
                # cross-lingual
                # loss_itc += self.get_contrastive_loss(text_feat_s, text_feat, idx=idx)

        # loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx=idx)

        return loss_itc_vs, loss_itc_vt, loss_itc_st


