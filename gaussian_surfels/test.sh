CUDA_VISIBLE_DEVICES=0 
root=$1
source /home/veily/anaconda3/etc/profile.d/conda.sh
conda init
conda activate dust3r
python ./dust3r/run.py --root_path "$root"
conda deactivate
conda activate gaussian_surfels2
python ./DSINE/test.py --datapath "$root"
python ./gaussian_surfels/train.py -s "$root"
python ./gaussian_surfels/render.py -m ./gaussian_surfels/outputs/ --img --depth 10
echo "Running test.sh with number"
cp ./gaussian_surfels/outputs/poisson_mesh_10_post.ply "$root/mesh.ply"
rm -rf ./gaussian_surfels/outputs/*
conda deactivate