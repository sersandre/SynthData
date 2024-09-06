import torch
import os
import TrainingFramework
import numpy
import cv2
import imageio.v2 as imageio
import open3d as o3d
from PIL import Image, ImageDraw
from tqdm import tqdm
from alive_progress import alive_bar


PATH = "/mnt/logicNAS/DataSets/synthetic_data/Models/MVE"

def main(scale=False):
    #model = torch.load("/mnt/logicNAS/PretrainedWeights/Generators/segmentor/magna.plt")
    #model.disable_resize()
    MASKPATH = "/home/sersandr/synthetic_data/nvdiffrec/workspace/images_rescaled/masks"

    for file in os.listdir(MASKPATH):
        print(file)
        mask = cv2.imread(os.path.join(MASKPATH, file))
        rescaled_mask = cv2.resize(mask, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(MASKPATH, file), rescaled_mask)




if __name__ == "__main__":
    os.chdir("/home/sersandr/synthetic_data/instant-ngp")
    size = Image.open(os.path.join("workspace/images", os.listdir("workspace/images")[0])).size
    print(size)