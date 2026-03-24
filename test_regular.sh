################ RESIDE
CUDA_VISIBLE_DEVICES=0 python infer.py --calc_metrics --model_depth 16 --vae_ckpt 'ckpts/vae_restorevar.ckpt' \
--var_ckpt 'ckpts/restorevar.pth'  --patch_nums 1 2 3 4 6 9 13 18 24 32 --json_path 'test_jsons/RESIDE.json' \
--result_dir 'results/'

############### Snow100k
CUDA_VISIBLE_DEVICES=0 python infer.py --calc_metrics --model_depth 16 --vae_ckpt 'ckpts/vae_restorevar.ckpt' \
--var_ckpt 'ckpts/restorevar.pth'  --patch_nums 1 2 3 4 6 9 13 18 24 32 --json_path 'test_jsons/Snow100k.json' \
--result_dir 'results/'

############### Rain13K
CUDA_VISIBLE_DEVICES=0 python infer.py --calc_metrics --model_depth 16 --vae_ckpt 'ckpts/vae_restorevar.ckpt' \
--var_ckpt 'ckpts/restorevar.pth'  --patch_nums 1 2 3 4 6 9 13 18 24 32 --json_path 'test_jsons/Rain13K.json' \
--result_dir 'results/'

############### LOL
CUDA_VISIBLE_DEVICES=0 python infer.py --calc_metrics --model_depth 16 --vae_ckpt 'ckpts/vae_restorevar.ckpt' \
--var_ckpt 'ckpts/restorevar.pth'  --patch_nums 1 2 3 4 6 9 13 18 24 32 --json_path 'test_jsons/LOLv1.json' \
--result_dir 'results/'

############### GoPro
CUDA_VISIBLE_DEVICES=0 python infer.py --calc_metrics --model_depth 16 --vae_ckpt 'ckpts/vae_restorevar.ckpt' \
--var_ckpt 'ckpts/restorevar.pth'  --patch_nums 1 2 3 4 6 9 13 18 24 32 --json_path 'test_jsons/GoPro.json' \
--result_dir 'results/'
