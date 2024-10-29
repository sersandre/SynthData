import argparse
import os
import sys
import json
import nvidia_smi
import time
import numpy
import imageio.v2 as imageio
import cv2
from PIL import Image
from alive_progress import alive_bar


DEFAULTDIR = "/home/sersandr/synthetic_data/Pipeline"
INSTANTNGP = "/home/sersandr/synthetic_data/instant-ngp"
COLMAP = "/home/sersandr/synthetic_data/colmap"
NVDIFFREC = "/home/sersandr/synthetic_data/nvdiffrec"
NVDIFFRECMC = "/home/sersandr/synthetic_data/nvdiffrecmc"

POISSON = "/home/sersandr/synthetic_data/PoissonRecon/Bin/Linux"
MVE = "/home/sersandr/synthetic_data/mve"
MVSTEXTURING = "/home/sersandr/synthetic_data/mvs-texturing"

DATASETS = "/mnt/logicNAS/DataSets/synthetic_data/Datasets"


METHODS = set(["NVDIFFREC", "NVDIFFRECMC", "INSTANTNGP", "COLMAP", "MVE"])

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--methods", nargs="+", type=str, default=["NVDIFFREC", "NVDIFFRECMC"])
    parser.add_argument("--data", nargs="+", type=str, default=[])
    parser.add_argument("--all", action="store_true")

    return parser.parse_args()

def load_config(args, method):
    with open(f"Configs/{method}.json") as conf:
        config = json.load(conf)
    return config

def pose_extraction(args, dataset):
    os.chdir(COLMAP)
    
    SCENEDIR = "workspace"
    IMDIR = f"{DATASETS}/{dataset}"
    FILE_COUNT = len(os.listdir(IMDIR))
    SEQ = 0

    while True:
        print("===== cleaning workspace =====")
        os.system("rm -r workspace")
        os.makedirs("workspace", exist_ok=False)
        cmd = f"python {DEFAULTDIR}/pose_extraction.py --scenedir {SCENEDIR} --image_path {IMDIR} --sequential_matcher {SEQ} --convert_model 1"
        os.system(cmd)
        with open(f"{DEFAULTDIR}/buffer.json", "r") as input:
            IMAGE_COUNT = json.load(input)
        if IMAGE_COUNT == FILE_COUNT:
            break
        print("===== COLMAP failed to register all provided images, re-running with adapted matching =====")
        SEQ = 1
    
    os.chdir(DEFAULTDIR)

def run_instant_ngp(args, dataset):

    IMAGEPATH = os.path.join(DATASETS, dataset)
    TEXTPATH = os.path.join(COLMAP, "workspace", "text")
    OUTPATH = os.path.join(INSTANTNGP, "workspace", "transforms.json")

    cmd = (f"python data_pipeline.py --image_path {IMAGEPATH} --text_path {TEXTPATH} --out_path {OUTPATH} --aabb_scale 4 --model_name {dataset} --add_poses")
    os.chdir(INSTANTNGP)
    assert os.getcwd() == INSTANTNGP, "Critical Error, current working directory not configured for running INSTANT-NGP"
    os.system(cmd)
    os.chdir(DEFAULTDIR)


def run_nvdiffrec(args, dataset):
    config = load_config(None, "nvdiffrec")

    #setting config
    MASK = config["generate_masks"]
    HEIC = int(os.listdir(os.path.join(DATASETS, dataset))[0].split(".")[-1] == ".heic")
    OUT = f"{config['out_dir']}/{dataset}"


    cmd = (f"python data_pipeline.py --image_input {os.path.join(DATASETS, dataset)} --heic {HEIC} --generate_masks {MASK} --resolution {config['training_res'][0]} {config['training_res'][1]}"
           f" --config {dataset}.json --batch_size {config['batch']} --mesh_scale {config['mesh_scale']} --iterations {config['iter']} --dmtet_scale {config['dmtet_grid']} --output_path {OUT}")
    os.chdir(NVDIFFREC)
    assert os.getcwd() == NVDIFFREC, "Critical Error, current working directory not configured for running NVDIFFREC"
    os.system(cmd)
    os.chdir(DEFAULTDIR)

def run_nvdiffrecmc(args, dataset):
    config = load_config(None, "nvdiffrecmc")

    MASK = config["generate_masks"]
    HEIC = int(os.listdir(os.path.join(DATASETS, dataset))[0].split(".")[-1] == ".heic")
    OUT = f"{config['out_dir']}/{dataset}"

    cmd = (f"python data_pipeline.py --image_input {os.path.join(DATASETS, dataset)} --heic {HEIC} --generate_masks {MASK} --resolution {config['training_res'][0]} {config['training_res'][1]}"
           f" --config {dataset}.json --batch_size {config['batch']} --mesh_scale {config['mesh_scale']} --iterations {config['iter']} --dmtet_scale {config['dmtet_grid']} --output_path {OUT}") 
    os.chdir(NVDIFFRECMC)
    assert os.getcwd() == NVDIFFRECMC, " Critical Error, current working directory not configured for running NVDIFFRECMC"
    os.system(cmd)
    os.chdir(DEFAULTDIR)


def run_mve_mvs(args, dataset):
    config = load_config(None, "mve")
    os.chdir(MVE)
    IMPATH = f"{DATASETS}/{dataset}"
    WORKSPACE = f"{MVE}/workspace"
    SCENEPATH = f"{WORKSPACE}/mveScene"

    ##################
    # mve functionaliy
    ##################

    MAKESCENE = "./apps/makescene/makescene"
    SFMRECON = "./apps/sfmrecon/sfmrecon"
    DMRECON = "./apps/dmrecon/dmrecon"
    SCENE2PSET = "./apps/scene2pset/scene2pset"
    FSSRECON = "./apps/fssrecon/fssrecon"
    MESHCLEAN = "./apps/meshclean/meshclean"

    ###################
    # mvs functionality
    ###################

    TEXRECON = "./build/apps/texrecon/texrecon"

    print("===== preparing workpace =====")
    os.system(f"rm -rf {WORKSPACE}")
    os.makedirs(WORKSPACE, exist_ok=False)
    os.makedirs(os.path.join(WORKSPACE, "mvsMesh"), exist_ok=False)
    os.makedirs(os.path.join(WORKSPACE, "masks"), exist_ok=False)
    os.makedirs(os.path.join(WORKSPACE, "images"))
    print("===== finished preparing workspace =====")

    assert os.getcwd() == MVE, "Critical Error, current directory not configured to run MVE"


    ##################
    # mask generation
    ##################
    if dataset.split("_")[-1] != "largeScene":
        import torch
        model = torch.load("/mnt/logicNAS/PretrainedWeights/Generators/segmentor/magna.plt")
        if not os.path.exists(os.path.join(WORKSPACE, "images")):
                os.makedirs(os.path.join(WORKSPACE, "images"))
        print(IMPATH)
        for file in os.listdir(IMPATH):
            print(f"moving file {file}")
            os.system(f"cp {IMPATH}/{file} {WORKSPACE}/images/{file}")

        IMPATH = os.path.join(WORKSPACE, "images")
    
        make_scene = f"{MAKESCENE} -i {IMPATH} {SCENEPATH}"
        os.system(make_scene)

        with alive_bar(len(os.listdir(IMPATH))) as bar:
            for file in os.listdir(os.path.join(SCENEPATH, "views")):
                image_path = os.path.join(SCENEPATH, "views", file, "original.jpg")
                print(f"===== generating mask for {file} =====")
                input_image = Image.open(image_path)
                res = input_image.size
                out = Image.fromarray((model({"x": input_image})["pred_0"]["cca"]["mask"])*255)
                mask_index = file.split(".")[0].split("_")[1]
                mask_path = os.path.join(SCENEPATH, "views", f"view_{mask_index}.mve", "mask.jpg")
                out.save(mask_path)

                _mask = cv2.imread(mask_path)
                rescaled_mask = cv2.resize(_mask, dsize=res, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(mask_path, rescaled_mask)
                bar()
    else:
        make_scene = f"{MAKESCENE} -i {IMPATH} {SCENEPATH}"
        os.system(make_scene)

    ################################
    # Dense point-cloud computation
    ################################

    sfm_recon = f"{SFMRECON} {SCENEPATH}"
    dm_recon = f"{DMRECON} -s0 {SCENEPATH}"
    scene_2_pset = f"{SCENE2PSET} -F0 {SCENEPATH} {SCENEPATH}/pset.ply"
    fss_recon = f"{FSSRECON} {SCENEPATH}/pset.ply {SCENEPATH}/surface.ply"
    mesh_clean = f"{MESHCLEAN} -t10 -c10000 {SCENEPATH}/surface.ply {SCENEPATH}/surface-clean.ply"
    cmds = [sfm_recon, dm_recon, scene_2_pset, fss_recon, mesh_clean]

    for cmd in cmds:
        os.system(cmd)
    
    os.chdir(MVSTEXTURING)
    assert os.getcwd() == MVSTEXTURING, "Critical Error, current directory not configured to run MVS-Texturing"
    OUTPATH = os.path.join(DEFAULTDIR, "out", "out_MVE", dataset)
    os.system(f"mkdir {OUTPATH}")

    mvs_texturing = f"{TEXRECON} {SCENEPATH}::undistorted {SCENEPATH}/surface-clean.ply {OUTPATH}/mvsMesh"
    os.system(mvs_texturing)
    os.chdir(DEFAULTDIR)

 
def run_colmap(args, dataset):
    os.chdir(COLMAP)

    IMPATH = f"{DATASETS}/{dataset}"
    DBPATH = f"workspace/database.db"
    SPARSE_PATH = f"workspace/sparse"
    DENSE_PATH = f"workspace/dense"
    TEXTPATH = f"workspace/text"

    print("===== preparing workspace ====")
    os.makedirs(f"{DENSE_PATH}", exist_ok=False)
    assert len(os.listdir(os.path.join(SPARSE_PATH, "0"))) > 0, "Critical Error, not all files for reconstruction available"
    print("===== finished preparing workspace =====")

    assert os.getcwd() == COLMAP, "Critical Error, current directory not configured to run COLMAP"

    #################################
    # Dense point-cloud computation
    #################################

    model_converter = f"colmap model_converter --input_path {SPARSE_PATH}/0 --output_path {TEXTPATH} --output_type CAM"
    image_undistorter = (f"colmap image_undistorter --image_path {IMPATH} --input_path {SPARSE_PATH}/0 --output_path {DENSE_PATH} --output_type COLMAP")
    patch_match_stereo = f"colmap patch_match_stereo --workspace_path {DENSE_PATH} --workspace_format COLMAP --PatchMatchStereo.geom_consistency true"
    stereo_fusion = f"colmap stereo_fusion --workspace_path {DENSE_PATH}  --workspace_format COLMAP  --input_type geometric --output_path {DENSE_PATH}/fused.ply"
    poisson_recon = f"./PoissonRecon --in /home/sersandr/synthetic_data/colmap/workspace/dense/fused.ply --out {DEFAULTDIR}/out/out_COLMAP/{dataset}.ply --depth 10 --pointWeight 0"

    
    os.system(model_converter)
    os.system(image_undistorter)
    os.system(patch_match_stereo)
    os.system(stereo_fusion)
    
    ##################################
    # mesh reconstuction
    ##################################
    os.chdir(POISSON)
    assert os.getcwd() == POISSON, "Critical Error, current directory not configured to run Poission reconstruction"
    os.system(poisson_recon)

    ##################################
    # texturing
    ##################################

    os.chdir(DEFAULTDIR)

def main():
    args = parse_args()
    DATA = set(args.data)

    for method in args.methods:
        assert method in METHODS, "invalid method: {method}, consider using: {METHODS}"

    for dataset in os.listdir(DATASETS):
        if dataset not in DATA and not args.all:
            continue
        os.system("rm -r buffer.json")

        nvidia_smi.nvmlInit()
        deviceCount = nvidia_smi.nvmlDeviceGetCount()
        os.system("echo $CUDA_VISIBLE_DEVICES")

        for device_index in range(deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_index)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            if info.free/info.used > 0.8:
                os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_index}"
                print(f"==== selecting GPU: {device_index} for pose_extraction")
                break
        os.system("echo $CUDA_VISIBLE_DEVICES")
        if args.methods != ["MVE"]:
            pose_extraction(args, dataset)
        nvidia_smi.nvmlShutdown()

        import torch

        for method in args.methods:
            os.makedirs(os.path.join(DEFAULTDIR, "out", f"out_{method}"), exist_ok=True)
            if method == "NVDIFFREC":
                print("running NVDIFFREC:")
                run_nvdiffrec(args, dataset)
            elif method == "INSTANTNGP":
                print("running INSTANT-NGP:")
                run_instant_ngp(args, dataset)
            elif method == "NVDIFFRECMC":
                print("running NVDIFFRECMC:")
                run_nvdiffrecmc(args, dataset)
            elif method == "COLMAP":
                print("running COLMAP:")
                run_colmap(args, dataset)
            elif method == "MVE":
                print("running MVE")
                run_mve_mvs(args, dataset)

if __name__ == "__main__":
    main()





