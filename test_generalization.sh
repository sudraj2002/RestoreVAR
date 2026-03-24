######### LHP
CUDA_VISIBLE_DEVICES=1 python infer.py --model_depth 16 --vae_ckpt 'ckpts/vae_restorevar.ckpt' \
--var_ckpt 'ckpts/restorevar.pth'  --patch_nums 1 2 3 4 6 9 13 18 24 32 --json_path 'test_jsons/LHP_test.json' \
--result_dir 'results_gen/'

######### REVIDE
CUDA_VISIBLE_DEVICES=1 python infer.py --model_depth 16 --vae_ckpt 'ckpts/vae_restorevar.ckpt' \
--var_ckpt 'ckpts/restorevar.pth'  --patch_nums 1 2 3 4 6 9 13 18 24 32 --json_path 'test_jsons/REVIDE.json' \
--result_dir 'results_gen/'

######### TOLED
CUDA_VISIBLE_DEVICES=1 python infer.py --model_depth 16 --vae_ckpt 'ckpts/vae_restorevar.ckpt' \
--var_ckpt 'ckpts/restorevar.pth'  --patch_nums 1 2 3 4 6 9 13 18 24 32 --json_path 'test_jsons/toled_test.json' \
--result_dir 'results_gen/'

######### POLED
CUDA_VISIBLE_DEVICES=1 python infer.py --model_depth 16 --vae_ckpt 'ckpts/vae_restorevar.ckpt' \
--var_ckpt 'ckpts/restorevar.pth'  --patch_nums 1 2 3 4 6 9 13 18 24 32 --json_path 'test_jsons/poled_test.json' \
--result_dir 'results_gen/'

######### LOLBlur (low-light + blur)
CUDA_VISIBLE_DEVICES=1 python infer.py --model_depth 16 --vae_ckpt 'ckpts/vae_restorevar.ckpt' \
--var_ckpt 'ckpts/restorevar.pth'  --patch_nums 1 2 3 4 6 9 13 18 24 32 --json_path 'test_jsons/LOLBlur.json' \
--result_dir 'results_gen/'

######### CDD (Haze + Rain)
CUDA_VISIBLE_DEVICES=1 python infer.py --model_depth 16 --vae_ckpt 'ckpts/vae_restorevar.ckpt' \
--var_ckpt 'ckpts/restorevar.pth'  --patch_nums 1 2 3 4 6 9 13 18 24 32 --json_path 'test_jsons/CDD_haze_rain.json' \
--result_dir 'results_gen/'

