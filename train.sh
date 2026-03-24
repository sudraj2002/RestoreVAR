torchrun --nproc_per_node=2 \
 --master_port=10245 train.py --patch_nums 1 2 3 4 6 9 13 18 24 32 --pn '1_2_3_4_6_9_13_18_24_32' \
 --exp_name RestoreVAR_train_VAR \
 --depth=16 --bs=48 --wp=0.0 --wp0=1.0 --tblr=0.000533333 --ep=50 --fp16=1 --alng=1e-3 \
 --data_path trainset.json --data_path_val valset.json --var_ckpt 'ckpts/var_d16.pth' \
 --vae_ckpt 'ckpts/vae_ch160v4096z32.pth'