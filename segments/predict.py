from typing import Tuple
import torch
import numpy as np
from tqdm import tqdm
import cv2 as cv

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.point_rend import add_pointrend_config


def list_to_ndarray(images):
    n_images = len(images)
    if not isinstance(images, np.ndarray):
        img = images[0]
        #彩色图像
        if len(img.shape) == 3:
            h, w, d = img.shape
            Images = np.zeros((n_images, h, w, d), dtype=np.uint8)
        #灰度图像
        elif len(img.shape) == 2:
            h, w = img.shape
            Images = np.zeros((n_images, h, w), dtype=np.uint8)
        for i in range(n_images):
            Images[i] = images[i]
    else:
        Images = images.copy()
    return Images

class Predictor(DefaultPredictor):
    def __call__(self, original_images):
        original_images = list_to_ndarray(original_images)
        with torch.no_grad():
            inputs = []
            for original_image in original_images:
                if self.input_format == "RGB":
                    #RGB 变为 BGR，适应opencv颜色通道
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = self.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

                input_ = {"image": image, "height": height, "width": width}
                inputs.append(input_)
            predictions = self.model(inputs)
            return predictions

class Predict(object):
    def __init__(self, yaml_path, model_path):
        cfg = get_cfg()
        add_pointrend_config(cfg)
        cfg.merge_from_file(yaml_path)

        cfg.MODEL.WEIGHTS = model_path
        # print("加载模型==> ",cfg.MODEL.WEIGHTS)
        
        cfg.MODEL.MASK_ON = True
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 ###
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2
        cfg.MODEL.POINT_HEAD.NUM_CLASSES = 2

        self.predictor = Predictor(cfg)

    def __call__(self, images, batch_size=4):
        images = list_to_ndarray(images)
        masks = np.zeros(images.shape[:3], dtype=np.uint8)

        for i in range(0, len(images), batch_size):
            inputs = images[i:i+batch_size]
            outputs = self.predictor(inputs)

            for j, output_ in enumerate(outputs):
                pred_mask = output_["sem_seg"].cpu().numpy().transpose(1, 2, 0)
                pred_mask = np.argmax(pred_mask, axis=2).astype(np.uint8)*255
                masks[i+j] = pred_mask
        
        torch.cuda.empty_cache()
        return  masks


if __name__=="__main__":
    yaml_path = "/media/veilytech/disk1/Code/permuto_sdf/child_work/FeetPoint/segments/projects/PointRend/configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml"
    model_path = "/media/veilytech/disk1/Code/permuto_sdf/child_work/FeetPoint/segments/model/model_final.pth"
    
    from pathlib import Path
    images_dir = "/media/veilytech/work/Jupyter/tools/camera_color_uniform/version2/valid/01/c_images"
    
    names = []
    images = []

    for pth in Path(images_dir).glob("*.*"):
        names.append(pth.stem)
        img = cv.imread(str(pth))
        images.append(img)
    
    pred = Predict(yaml_path, model_path)
    masks = pred(images, batch_size=4)
    
    n_masks = masks.shape[0]
    for i in range(n_masks):
        mk_pth = Path(
            "/media/veilytech/work/Jupyter/tools/camera_color_uniform/version2/valid/01/u_masks", names[i]+".png")
        cv.imwrite(str(mk_pth), masks[i])

