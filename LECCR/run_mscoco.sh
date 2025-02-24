# sleep 5h
# 4,5,6,7
# 0,1,2,3
# 0,1,2,3,4,5,6,7
# --log_ps
#  /mnt/workspace/CCR2/xlmr/data/cclm_3m_epoch_29.th
# sleep 4h
export MKL_SERVICE_FORCE_INTEL=1

# CUDA_VISIBLE_DEVICES=2,7 python3 run.py --dist f2 --task itr_multi30k --config configs/cclm-base-ft/Retrieval_multi30k_fr_ft.yaml --output_dir outputs/multi30k/fr/full_2e-5_mean --bs 128 --seed 42 --epoch 30 --checkpoint null
#  checkpoint_best.pth  queries_4_query_layer4_caption_loss_0.05_with_sim
# CUDA_VISIBLE_DEVICES=1,3 python3 run.py --dist j2 --task itr_multi30k_caption --config configs/cclm-base-ft/Retrieval_coco_zh_ft.yaml --output_dir outputs/mscoco/zh/queries_4_query_calayer3_caption_loss_0.07_reg_new_0.01_dstl_0.5_cvloss_0.01_dualcross_layer2 --bs 128 --seed 42 --epoch 50 --checkpoint null
# sleep 10s
CUDA_VISIBLE_DEVICES=6,7 python3 run.py --dist l2 --task itr_multi30k_caption --config configs/cclm-base-ft/Retrieval_coco_zh_ft.yaml --output_dir outputs/mscoco/zh/test2 --bs 128 --seed 42 --epoch 70 --checkpoint null