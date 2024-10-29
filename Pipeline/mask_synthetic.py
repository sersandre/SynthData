import torch
import os
import TrainingFramework
import numpy
import cv2
import argparse
import imageio.v2 as imageio
import open3d as o3d
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from alive_progress import alive_bar

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method")

    return parser.parse_args()

def convert_images(args):
    root = "/mnt/logicNAS/DataSets/synthetic_data/Rendering"
    method = args.method


    for dataset in os.listdir(os.path.join(root, method)):
        images = os.listdir(os.path.join(root, method, dataset))
        for image in images:
            path = os.path.join(root, method, dataset, image)
            rgba_image = Image.open(path)
            if rgba_image.mode != "RGB":
                rgb_image = rgba_image.convert("RGB")
                print(image, dataset)
                rgb_image.save(os.path.join(root, method, dataset, image))
            else:
                print("skipping alpha conversion")

def split(args):
    root = "/mnt/logicNAS/DataSets/synthetic_data/Rendering"
    method = args.method


    for dataset in os.listdir(os.path.join(root, method)):
        images = os.listdir(os.path.join(root, method, dataset))

        count = len(images)
        indices = np.array([i for i in range(count)])
        np.random.shuffle(indices)

        train, test = set(indices[:int(0.8*count)]), set(indices[int(0.8*count):])

        os.makedirs(os.path.join(root, method, dataset, "train"))
        os.makedirs(os.path.join(root, method, dataset, "valid"))
        

        for index, image in enumerate(images):
            print(f"moving {image}, {dataset}")
            if index in train:
                os.system(f"mv {os.path.join(root, method, dataset, image)} {os.path.join(root, method, dataset, 'train', image)}")
            else:
                os.system(f"mv {os.path.join(root, method, dataset, image)} {os.path.join(root, method, dataset, 'valid', image)}")

def generate_masks(args):
    model = torch.load("/mnt/logicNAS/PretrainedWeights/Generators/segmentor/magna.plt")
    model.disable_resize()
    root = "/mnt/logicNAS/DataSets/synthetic_data/Rendering"
    method = args.method

    for dataset in os.listdir(os.path.join(root, method)):
        for image in os.listdir(os.path.join(root, method, dataset, "train")):
            if "mask_rgb" in image:
                print("skipping...")
                continue
            input_image = Image.open(os.path.join(root, method, dataset, "train", image))
            print(f"generating mask for {image}, {dataset}")
            out = Image.fromarray((model({"x": input_image})["pred_0"]["cca"]["mask"])*255)
            filename = image.split(".")[0]

            out.save(os.path.join(root, method, dataset, "train", f"{filename}_mask_rgb.png"))
        
        for image in os.listdir(os.path.join(root, method, dataset, "valid")):
            if "mask_rgb" in image:
                print("skipping...")
                continue
            input_image = Image.open(os.path.join(root, method, dataset, "valid", image))
            print(f"generatin mask for {image}, {dataset}")
            out = Image.fromarray((model({"x": input_image})["pred_0"]["cca"]["mask"])*255)
            filename = image.split(".")[0]

            out.save(os.path.join(root, method, dataset, "valid", f"{filename}_mask_rgb.png"))


def rename_failes(args):
    root = "/mnt/logicNAS/DataSets/synthetic_data/Rendering"
    method = args.method
    for dataset in os.listdir(os.path.join(root, method)):
        for dir in os.listdir(os.path.join(root, method, dataset)):
            if dir in ["train", "valid"]:
                print("all good, skipping...")
            else:
                os.rename(os.path.join(os.path.join(root, method, dataset, dir)), os.path.join(root, method, dataset, "valid"))
                print("renaming {}...".format(dir))

if __name__ == "__main__":
    args = parse_args()
    #rename_failes(args)
    convert_images(args)
    split(args)
    generate_masks(args)
    #rename_failes(args)
    #convert_images(args)
    #split(args)
    
       

