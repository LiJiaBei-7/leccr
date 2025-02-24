
export MKL_SERVICE_FORCE_INTEL=1


CUDA_VISIBLE_DEVICES=4,5 python3 run.py --dist f2 --task itr_video_caption --config configs/cclm-base-ft/Retrieval_msrvtt.yaml --output_dir outputs/msrvtt/full --bs 128 --seed 42 --epoch 50 --checkpoint null 
