#! /bin/bash
# source /home/veily/anaconda3/condabin/conda
source /home/veily/anaconda3/bin/activate feet3d

# conda activate feet3d
# python3 /home/veily/Feet3D/aiserver.py

pm2 start /home/veily/Feet3D/aiserver.py --interpreter /home/veily/anaconda3/envs/feet3d/bin/python3