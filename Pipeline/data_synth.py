import os
import sys
import shutil
import argparse
import cv2
import io
import imageio
import json
import common 
import numpy as np

from PIL import Image
from tqdm import tqdm
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngp_root", type=Path, default="/home/sersandr/synthetic_data/instant-ngp")
    parser.add_argument("--instant_ngp", action="store_true")
    parser.add_argument("--nvdiffrec", action="store_true")
    parser.add_argument("--nvdiffremc", action="store_true")
    parser.add_argument("--mve", action="store_true")
    parser.add_argument("--object")

    return parser.parse_args()

def linear_to_srgb(img):
	limit = 0.0031308
	return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)

def write_image_imageio(img_file, img, quality):
	img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
	kwargs = {}
	if os.path.splitext(img_file)[1].lower() in [".jpg", ".jpeg"]:
		if img.ndim >= 3 and img.shape[2] > 3:
			img = img[:,:,:3]
		kwargs["quality"] = quality
		kwargs["subsampling"] = 0
	imageio.imwrite(img_file, img, **kwargs)
      
def write_depth_npz(
    filename: Path, depth: np.ndarray, alpha: np.ndarray, use_fp16: bool = True
) -> None:
    """Saves a depth image to an NPZ file.
    :param filename: Path to the output depth image file.
    :param im: ndarray with the depth image to save.
    :param alpha: ndarray with the alpha channel of the depth image to save.
    :param use_fp16: Whether to use FP16 for depth and alpha channels.
    """
    if filename.suffix.lower() != ".npz":
        raise ValueError("Only NPZ format is currently supported.")
    if use_fp16:
        depth = depth.astype(np.float16)
        alpha = alpha.astype(np.float16)
    np.savez_compressed(filename, depth=depth, alpha=alpha)

def synth_nerf(args, exposure=0):
    SNAPSHOT = f"/home/sersandr/synthetic_data/Pipeline/out/out_INSTANTNGP/{args.object}.ingp"
    OUT = "/home/sersandr/synthetic_data/Pipeline/SynthData"


    sys.path.append(str(args.ngp_root / "build"))
    import pyngp as ngp

    testbed = ngp.Testbed()
    testbed.load_snapshot(SNAPSHOT)

    testbed.background_color = [0.0, 0.0, 0.0, 0.0]
    testbed.render_mode = ngp.RenderMode.Shade
    testbed.nerf.render_gbuffer_hard_edges = True

    if os.path.exists(OUT) and os.listdir(OUT):
        print("clearing output directory")
        os.system(f"rm -rf {OUT}")
        os.makedirs(OUT, exist_ok=False)

    with open("../instant-ngp/workspace/transforms.json", "r") as f:
        config = json.load(f)
        frames = config["frames"]

    for idx, frame in enumerate(frames):

        cam_matrix = np.matrix(frame["transform_matrix"])
        cam_matrix = cam_matrix[0:3, :]
        print(cam_matrix)
        testbed.set_nerf_camera_matrix(cam_matrix)
        resolution = np.array([3024, 3024])
        output = testbed.render(*resolution, 1, True)
        
        common.write_image(f"../Pipeline/SynthData/view_{idx}.jpg", np.clip(output * 2** exposure, 0.0, 1.0), quality=100)



if __name__ == "__main__":
    args = parse_args()
    if args.instant_ngp:
        synth_nerf(args)
    
