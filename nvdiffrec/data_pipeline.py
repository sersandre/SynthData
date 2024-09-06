import os 
import shutil
import sys
import glob
import numpy 
import argparse
import cv2
import imageio.v2 as imageio
import torch
import json
from PIL import Image
import TrainingFramework

PILJPEG = "<class, PIL.JpegImagePlugin.JpegImageFile>"
PILIMAGE = "PIL.ImageFile"
PIPELINE = "/home/sersandr/synthetic_data/Pipeline"
HOMEIDR = "/home/sersandr/synthetic_data/nvdiffrec"

def parse_args():
    parser = argparse.ArgumentParser()

    # Arguments for workspace setup
    parser.add_argument("--scenedir", default="workspace", help="sets scene directory for COLMAP and nvdiffrec Pipeline")
    parser.add_argument("--image_path", default="images", help="sets image directory within scenedir")
    parser.add_argument("--mask_path", default="masks", help="sets mask directory within scenedir")
    # Arguments for COLMAP posegeneration
    parser.add_argument("--sequential_matcher", type=int, default=0, help="toggle sequential matching for COLMAP pipeline")
    parser.add_argument("--convert_modell", type=int, default=0, help="convert COLMAP sparse recontruction to .json format for Instant-NGP sanity check")
    parser.add_argument("--aabb_scale", type=int, default=32, choices=[1, 2, 4, 8, 16, 32, 64, 128], help="Large scene scale factor")
    parser.add_argument("--json", type=int, default=0, help="set to 1 to generate transforms.json file for Instant-NGP setup")
    # Arguments for Datapipeline
    parser.add_argument("--image_input", default="", help="sets path to directory containing input images")
    parser.add_argument("--mask_input", default="", help="sets path to directory containing masks")
    parser.add_argument("--heic", type=int, default=0, help="specify .heic input format")
    parser.add_argument("--generate_masks", type=int, default=0, help="enable mask generation using MVIP model")
    # Arguments for training config
    parser.add_argument("--resolution", nargs="+", type=int, default=[512, 512], help="resolution for nvdiffrec pipeline")
    parser.add_argument("--config", default="config.json", help="config name")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for nvdiffrec pipeline, high batch sizes can lead to substential memory usage")
    parser.add_argument("--mesh_scale", type=float, default=2.5, help="mesh scale for nvdiffrec pipeline")
    parser.add_argument("--iterations", type=int, default=5000, help="sets iteration count for nvdiffrec pipeline")
    parser.add_argument("--dmtet_scale", type=int, default=64, choices=[64, 128], help="sets scale of underlying dmtet mesh")
    parser.add_argument("--output_path", default="", help="sets output directory for final mesh as well as checkpoint renderings")

    args = parser.parse_args()
    return args

def clean_working_dir(args, *exceptions):
    '''
    clears working directory for next cycle of pipeline
    '''
    SCENEDIR = args.scenedir
    print(SCENEDIR)
    assert os.path.exists(SCENEDIR), f"FATAL: {SCENEDIR} is no valid directory"
    print("===== cleaning working directory =====")
    for file in os.listdir(SCENEDIR):
        if file not in exceptions:
            if os.path.isdir(os.path.join(SCENEDIR, file)):
                shutil.rmtree(os.path.join(SCENEDIR, file))
            else:
                os.remove(os.path.join(SCENEDIR, file))

def gen_config(args):
    '''
    generates config for nvdiffrec with specs provided by arguments
    '''
    SCENEDIR = args.scenedir
    ITER = args.iterations
    RES = args.resolution
    BATCH = args.batch_size
    SCALE = args.mesh_scale
    REFMESH = os.path.join(SCENEDIR, 'images_rescaled')
    OUTDIR = args.output_path
    NAME = args.config
    DMTET = args.dmtet_scale
    print(f"==== generating traing config: {NAME} ====")
    json_out = {
    "ref_mesh": REFMESH,
    "random_textures": True,
    "isosurface" : "flexicubes",
    "iter": ITER,
    "save_interval": 100,
    "texture_res": [ 2048, 2048 ],
    "train_res": RES,
    "batch": BATCH,
    "learning_rate": [0.03, 0.03],
    "kd_min" : [0.03, 0.03, 0.03],
    "kd_max" : [0.8, 0.8, 0.8],
    "ks_min" : [0, 0.08, 0],
    "ks_max" : [0, 1.0, 1.0],
    "dmtet_grid" : DMTET,
    "mesh_scale" : SCALE,
    "camera_space_light" : True,
    "background" : "white",
    "display" : [{"bsdf":"kd"}, {"bsdf":"ks"}, {"bsdf" : "normal"}],
    "out_dir": OUTDIR
    }      

    with open(f"./configs/{NAME}", "w") as outfile:
        json.dump(json_out, outfile, indent=4)

def create_imdict(image_path, pattern, dict):
    '''
    recursive method to get all image files with specific pattern from nested directory
    '''
    if os.path.isdir(image_path):
        for dir in os.listdir(image_path):
            create_imdict(os.path.join(image_path, dir), pattern, dict)
    else:
        file_name = os.path.basename(image_path)
        if issubsequence(pattern, file_name):
            dict["data"][file_name] = image_path
    return dict
    

def create_maskdict(mask_path, pattern, dict):
    '''
    recursive method to get all mask files with specific pattern from nested directory
    '''
    if os.path.isdir(mask_path):
        for dir in os.listdir(mask_path):
            create_maskdict(os.path.join(mask_path, dir), pattern, dict)
    else:
        file_name = mask_path.split("/")[-1]
        if issubsequence(pattern, file_name):
            dict["data"][file_name] = Image.open(f"{mask_path}")
    return dict

def convert_heic(image_path, imdict):
    '''
    converts .heic images (e.g. Iphone images ) to jpg format readable by COLMAP
    '''
    IMPATH = image_path
    imdict["data_type"] = str
    os.makedirs(f"{IMPATH}/convert", exist_ok=True)
    counter = 1
    for image in os.listdir(IMPATH):
        if not os.path.isdir(f"{IMPATH}/{image}"):
            cmd = f"heif-convert -q 100 {IMPATH}/{image} {IMPATH}/convert/image_{counter}.jpg"
            os.system(cmd)
            imdict["data"][f"image_{counter}.jpg"] = f"{IMPATH}/convert/image_{counter}.jpg"
            counter += 1
    return imdict

def load_data(args, *dicts):
    '''
    load data in working directory, generate masks if necessary, scale images/masks to resolution set in config file
    '''
    clean_working_dir(args)
    SCENEDIR = args.scenedir
    IMDIR = args.image_path
    RES = args.resolution
    rescaled_path = os.path.join(SCENEDIR, IMDIR + "_rescaled")
    os.makedirs(rescaled_path, exist_ok=True)
    os.makedirs(os.path.join(rescaled_path, IMDIR))

    for dictionary in dicts:
        # loads image and mask data in set scene directory
        assert len(dictionary) > 0, "empty input provided : {}".format(dictionary)
        data_type = dictionary["data_type"]
        PATH = os.path.join(args.scenedir, dictionary["folder"])
        os.makedirs(PATH, exist_ok=True)

        if data_type == str:
            for file in dictionary["data"].values():
                shutil.copyfile(file, os.path.join(PATH, os.path.basename(file)))
        elif data_type == PILJPEG or data_type == PILIMAGE:
            for file in dictionary["data"].keys():
                image = dictionary["data"][file]
                image.save(os.path.join(PATH, file))
        else:
            raise TypeError("{!r} is no valid datatype, possbile datatypes are: {!r}, {!r}, {!r}".format(data_type, str, PILJPEG, PILIMAGE))

    for file in os.listdir(os.path.join(SCENEDIR, IMDIR)):
        # rescales images to fit resolution set in config file
        image = torch.tensor(imageio.imread(os.path.join(SCENEDIR, IMDIR, file)).astype(numpy.float32) / 255.0)
        image = image[None, ...].permute(0, 3, 1, 2)
        rescaled_image = torch.nn.functional.interpolate(image, RES, mode = "area")
        rescaled_image = rescaled_image.permute(0, 2, 3, 1)[0, ...]
        outfile = os.path.join(rescaled_path,"images", os.path.basename(file))
        imageio.imwrite(outfile, numpy.clip(numpy.rint(rescaled_image.numpy() * 255), 0, 255).astype(numpy.uint8))

    if args.generate_masks:
        # generated masks for provided images using Fraunhofer MVIP segmentation model
        model =  torch.load("/mnt/logicNAS/PretrainedWeights/Generators/segmentor/magna.plt")
        MASKDIR = os.path.join(rescaled_path, "masks")
        IMDIR = os.path.join(SCENEDIR, IMDIR)
        os.makedirs(MASKDIR)
        if RES != [512, 512]:
            model.disable_resize()
            IMDIR = os.path.join(rescaled_path, "images")
        for file in os.listdir(IMDIR):
            print(f"==== generating mask for {file} ====")
            input_image = Image.open(os.path.join(IMDIR, file))
            out = Image.fromarray((model({"x": input_image})["pred_0"]["cca"]["mask"])*255)
            out.save(os.path.join(MASKDIR, os.path.basename(file)))
        for file in os.listdir(MASKDIR):
            print(f"==== rescaling mask for: {file}")
            mask = cv2.imread(os.path.join(MASKDIR, file))
            rescaled_mask = cv2.resize(mask, dsize=(RES[0], RES[1]), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(MASKDIR, file), rescaled_mask)
    else:
        MASKPATH = os.path.join(rescaled_path, "masks")
        os.makedirs(MASKPATH, exist_ok=True)
        # rescales provided masks to fit resolution set in config file
        for file in os.listdir(os.path.join(SCENEDIR, args.mask_path)):
            mask = torch.tensor(imageio.imread(os.path.join(SCENEDIR, args.mask_path, file)).astype(numpy.float32) / 255.0)
            mask = mask[None, ...].permute(0, 3, 1, 2)
            rescaled_mask = torch.nn.functional.interpolate(mask, RES, mode = "area")
            rescaled_mask = rescaled_mask.permute(0, 2, 3, 1)[0, ...]
            outfile = os.path.join(rescaled_path, "masks", os.path.basename(file))
            imageio.imwrite(outfile, numpy.clip(numpy.rint(rescaled_mask.numpy() * 255), 0, 255).astype(numpy.uint8))

def issubsequence(pattern, str):
    '''
    helperfunction for pattern matching during image processing
    '''
    pattern = pattern.strip("*")
    return pattern.lower() in str.lower()

if __name__ == "__main__":
    while True:
        args = parse_args()
        IMAGEIN = args.image_input.rstrip("/")
        MASKIN = args.mask_input.rstrip("/")
        CONFIG = args.config
        if args.heic:
            imdict = convert_heic(IMAGEIN, {"folder": "images", "data_type": None, "data": {}})
        else:
            imdict = create_imdict(IMAGEIN, "*.jpg", {"folder": "images", "data_type": str, "data": {}})
        if args.generate_masks:
            load_data(args, imdict)
        else:
            maskdict = create_maskdict(MASKIN, ".JPG", {"folder": "masks", "data_type": PILJPEG, "data": {}})
            load_data(args, imdict, maskdict)
        gen_config(args)

        print("===== loading poses =====")
        shutil.copyfile("/home/sersandr/synthetic_data/colmap/workspace/poses_bounds.npy", os.path.join(args.scenedir, "images_rescaled", "poses_bounds.npy"))
        print("===== finished loading poses =====")

        proceed = input(f"finished running datapipeline, proceed to start training with current config? [y/n]").lower()
        if proceed == "y":
            os.system(f"docker run --gpus all -it --rm -v /home/sersandr/synthetic_data/nvdiffrec:/workspace -it nvdiffrec:v1 \
                       torchrun --nproc_per_node=2 train.py --config configs/{args.config} -o {args.output_path}")
            print("===== finished reconstruction, moving files to Pipeline =====")
            shutil.move(os.path.join(HOMEIDR, 'out', CONFIG.rstrip('.json')), os.path.join(PIPELINE, 'out', 'out_NVDIFFREC'))
            sys.exit(1)
        else:
            sys.exit(1)  


