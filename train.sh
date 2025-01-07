CUDA_VISIBLE_DEVICES='1,2,3,4' python -m torch.distributed.launch --nproc_per_node=4  --master_port 29500 --use_env \
main.py --batch_size 2 --epochs 30 --lr_drop 20 --sequence_length 32 --dataset_file 'mvp' \
--output_dir 'weights/bs2_32T_sz320_r50_4card_lr_2-4e_[20_30]'