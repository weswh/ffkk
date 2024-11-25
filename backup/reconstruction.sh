# UUID="202407210003"
# DATA_PATH="/home/veily/Feet3D/data/$UUID"

DATA_PATH=$1
source /home/chuck/anaconda3/etc/profile.d/conda.sh
conda init
conda activate gaussian_surfels
startTime=$(date +%s)

# # 运行MVS
# cp -r "$DATA_PATH/u_images/" /home/veily/Feet3D/CL-MVSNet/TMP/image
# cp -r "$DATA_PATH/u_masks/select/" /home/veily/Feet3D/CL-MVSNet/TMP/mask
# cp "$DATA_PATH/transforms.json" /home/veily/Feet3D/CL-MVSNet/TMP/transforms.json
# cp /home/veily/Feet3D/CL-MVSNet/pair.txt /home/veily/Feet3D/CL-MVSNet/TMP/pair.txt

# # python3 /home/veily/Feet3D/CL-MVSNet/rename.py --folder_path /home/veily/Feet3D/CL-MVSNet/TMP/image  # 重命名——相机数量少了的时候用
# # python3 /home/veily/Feet3D/CL-MVSNet/rename.py --folder_path /home/veily/Feet3D/CL-MVSNet/TMP/mask --label "mask"  # 重命名——相机数量少了的时候用

# python3 /home/veily/Feet3D/CL-MVSNet/nerf_to_cams.py --root_path /home/veily/Feet3D/CL-MVSNet/TMP  # json文件转换
CUDA_VISIBLE_DEVICES=0 python3  /home/chuck/gaussian_surfels/CL-MVSNet-master/main.py \
        --test \
        --dataset_name "general_eval"  \
        --datapath  "$DATA_PATH"   \
        --outdir  "$DATA_PATH"   \
        --img_size 512 640  \
        --resume /home/chuck/gaussian_surfels/CL-MVSNet-master/pretrained_model/model.ckpt  \
        --testlist /home/chuck/gaussian_surfels/CL-MVSNet-master/datasets/lists/dtu/test.txt  \
        --depth_thres 0.5  \
        --img_dist_thres 5  \
        --conf 0.5

# mv /home/veily/Feet3D/CL-MVSNet/TMP/mvs.ply "$DATA_PATH"
# rm -rf /home/veily/Feet3D/CL-MVSNet/TMP/*
# python3 /home/veily/Feet3D/CL-MVSNet/plus_normal.py --pcd_path "$DATA_PATH/mvs.ply"
# echo "Finish running MVS!"

# # 运行DSINE
# python3 /home/veily/Feet3D/DSINE/test.py --image_path "$DATA_PATH/u_images"
# echo "Finish running DSINE!"

# # 运行GaussianSurfels
# python3 /home/veily/Feet3D/gaussian_surfels/train.py -s "$DATA_PATH"
# python3 /home/veily/Feet3D/gaussian_surfels/render.py -m /home/veily/Feet3D/gaussian_surfels/output/tmp --img --depth 10
# cp /home/veily/Feet3D/gaussian_surfels/output/tmp/poisson_mesh_10_post.ply "$DATA_PATH/mesh.ply"
# rm -rf /home/veily/Feet3D/gaussian_surfels/output/tmp/*
# echo "Finish running GaussianSurfels!"

# endTime=$(date +%s)
# totalTime=$((endTime-startTime))
# echo "Use time: $totalTime"
