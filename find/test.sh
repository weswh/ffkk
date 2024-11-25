source /home/veily/anaconda3/etc/profile.d/conda.sh
conda init
conda activate find
cd FIND
python ./src/eval/eval_3d.py --exp_name 3D_only
cd ..
conda deactivate