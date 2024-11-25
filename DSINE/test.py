import os
import sys
import glob
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import utils.utils as utils
import argparse
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
#import os
#parser = ArgumentParser(description="Training script parameters")

#print(root_path)
def test_samples(args, model, intrins=None, device='cpu'):
    root_path = args.datapath
    #print(root_path)
    # 1.修改数据集的路径
    # 2.修改保存的路径
    img_paths = glob.glob(root_path+'/u_images/*.jpg')
    output_dir = os.path.join(root_path,'DSINE')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    img_paths.sort()
    # normalize
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    with torch.no_grad():
        for img_path in img_paths:
            ext = os.path.splitext(img_path)[1]
            img = Image.open(img_path).convert('RGB')
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
            _, _, orig_H, orig_W = img.shape
            # zero-pad the input image so that both the width and height are multiples of 32
            l, r, t, b = utils.pad_input(orig_H, orig_W)
            img = F.pad(img, (l, r, t, b), mode="constant", value=0.0)
            img = normalize(img)
            #读取相机参数
            intrins_path = img_path.replace(ext, '_cam.txt')
            # intrins_path = intrins_path.replace("images", "DSINE/intrinsic")
            intrins_path = intrins_path.replace("u_images", "cams")
            if os.path.exists(intrins_path):
                # NOTE: camera intrinsics should be given as a txt file
                # it should contain the values of fx, fy, cx, cy
                #print("加载相机内参{}".format(intrins_path))
                intrins = utils.get_intrins_from_txt(intrins_path, device=device).unsqueeze(0)
                # print("加载内参")
            else:
                # NOTE: if intrins is not given, we just assume that the principal point is at the center
                # and that the field-of-view is 60 degrees (feel free to modify this assumption)
                intrins = utils.get_intrins_from_fov(new_fov=60.0, H=orig_H, W=orig_W, device=device).unsqueeze(0)

            intrins[:, 0, 2] += l
            intrins[:, 1, 2] += t

            pred_norm = model(img, intrins=intrins)[-1]
            pred_norm = pred_norm[:, :, t:t + orig_H, l:l + orig_W]
            pred_norm[:, 0:3, :, :] *= -1      # []  # [b,c,w,h]
            # 保存npy文件
            #print("保存的npy通道顺序",pred_norm.shape)
            np.save(os.path.join(output_dir, img_path.split('/')[-1].replace('.jpg', '_normal.npy')),
                    pred_norm.cpu().detach().numpy().astype(np.float16))

            # save to output folder
            # NOTE: by saving the prediction as uint8 png format, you lose a lot of precision
            # if you want to use the predicted normals for downstream tasks, we recommend saving them as float32 NPY files
            # 保存成图像
            pred_norm_np = pred_norm.cpu().detach().numpy()[0, :, :, :].transpose(1, 2, 0)  # (W, H, 3)

            # trans = np.array([[-1., 0, 0], [0, -1., 0], [0, 0, -1]])
            # pred_norm_np = (trans @ pred_norm_np[..., None]).squeeze()

            pred_norm_rgb = ((pred_norm_np + 1.0) / 2.0 * 255.0).astype(np.uint8)  # to images
            # pred_norm_np = (255-((pred_norm_np + 1.0) / 2.0 * 255.0)).astype(np.uint8)  # to images
            target_path = os.path.join(output_dir, img_path.split('/')[-1])
            # print("save to {}".format(target_path.replace('.png',"_normal.png")))
            im = Image.fromarray(pred_norm_rgb)
            im.save(target_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='dsine', type=str, help='model checkpoint')
    parser.add_argument('--mode', default='samples', type=str, help='{samples}')
    # = argparse.ArgumentParser(description="CLMVSNet args")
    parser.add_argument("--datapath",
                    default="/media/veily105/ModelLjj/01-colleague/lyl/20240517/cy",
                    type=str)
    #args = parser.parse_args()
   
    args = parser.parse_args()

    # define model
    device = torch.device('cuda')

    from models.dsine import DSINE

    model = DSINE().to(device)
    model.pixel_coords = model.pixel_coords.to(device)
    model = utils.load_checkpoint('./DSINE/checkpoints/%s.pt' % args.ckpt, model)
    model.eval()

    if args.mode == 'samples':
        test_samples(args, model, intrins=None, device=device)
