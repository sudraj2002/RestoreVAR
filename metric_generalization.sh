conda create -y --name restorevar_eval1 --file spec-file.txt
conda run -n restorevar_eval1 pip install opencv-python==4.10.0.84
conda create --name restorevar_eval2 python=3.9 -y
conda run -n restorevar_eval2 pip install -r requirements_iqa.txt
conda run -n restorevar_eval1 python split_saved_imgs.py
conda run -n restorevar_eval1 python conv_png.py
conda run -n restorevar_eval2 python calc_metric.py