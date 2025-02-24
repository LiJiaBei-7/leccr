
export MKL_SERVICE_FORCE_INTEL=1


# train fr
CUDA_VISIBLE_DEVICES=6,7 python3 run.py --dist f2 --task itr_multi30k_caption --config configs/cclm-base-ft/Retrieval_multi30k_fr_ft.yaml --output_dir outputs/LECCR/multi30k/fr/full --bs 128 --seed 42 --epoch 70 --checkpoint null

# train de
# CUDA_VISIBLE_DEVICES=6,7 python3 run.py --dist f2 --task itr_multi30k_caption --config configs/cclm-base-ft/Retrieval_multi30k_de_ft.yaml --output_dir outputs/LECCR/multi30k/de/full --bs 128 --seed 42 --epoch 50 --checkpoint null

# train cs
# CUDA_VISIBLE_DEVICES=6,7 python3 run.py --dist f2 --task itr_multi30k_caption --config configs/cclm-base-ft/Retrieval_multi30k_cs_ft.yaml --output_dir outputs/LECCR/multi30k/cs/full --bs 128 --seed 42 --epoch 50 --checkpoint null


