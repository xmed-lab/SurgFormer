CUDA_VISIBLE_DEVICES='1,2,3,4' python -m torch.distributed.launch --nproc_per_node=4  --master_port 29500 --use_env \
main.py --batch_size 2 --epochs 30 --lr_drop 20 --sequence_length 32 --dataset_file 'mvp' \
--output_dir 'weights/197_bs2_32T_sz320_r50_4card_lr_2-4e_CE_coef1515_mvp_[20_30]'




#CUDA_VISIBLE_DEVICES='5,6' python -m torch.distributed.launch --nproc_per_node=2  --master_port 29500 --use_env \
#main.py --batch_size 2 --epochs 30 --lr_drop 20 --output_dir 'weights/bs2_32T_sz320_r50_2card_lr_2-4e_coef1515_loss_xT_30vids_4tasks' --backbone resnet50 --sequence_length 32