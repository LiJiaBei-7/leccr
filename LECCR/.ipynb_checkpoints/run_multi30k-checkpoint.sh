# sleep 5h
# 4,5,6,7
# 0,1,2,3
# 0,1,2,3,4,5,6,7
# --log_ps
#  /mnt/workspace/CCR2/xlmr/data/cclm_3m_epoch_29.th
# sleep 1h
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 run.py --dist f8 --task itr_multi30k --config configs/cclm-base-ft/Retrieval_multi30k_fr_ft.yaml --output_dir outputs/multi30k/fr/incoporate_caption --bs 128 --seed 42 --epoch 30 --checkpoint data/cclm_3m_epoch_29.th