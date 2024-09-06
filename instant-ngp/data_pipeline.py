import os
import sys
import torch
import TrainingFramework
import shutil
import argparse
import glob
import cv2 
import json
import math
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--skip_early", default=0, help="")
    parser.add_argument("--aabb_scale", default=32, choices=["1", "2", "4", "8", "16", "32", "64", "128"], help="")
    parser.add_argument("--image_path", required=True, help="")
    parser.add_argument("--text_path", required=True, help="")
    parser.add_argument("--out_path", required=True, help="")
    parser.add_argument("--keep_colmap_coords", action="store_true")
    parser.add_argument("--model_name", default="")
    parser.add_argument("--add_poses", action="store_true")
    parser.add_argument("--image_count", type=int, default=100)

    args = parser.parse_args()
    return args

def run_instantngp(args):
    MODELNAME = args.model_name
    IMAGEDIR = f"workspace/images"
    MASKDIR = f"workspace/masks"
    os.system("rm -r workspace")
    os.makedirs("workspace", exist_ok=False)
    os.makedirs(IMAGEDIR, exist_ok=False)
    os.makedirs(MASKDIR, exist_ok=False)

    model =  torch.load("/mnt/logicNAS/PretrainedWeights/Generators/segmentor/magna.plt")

    resolution = []
    convert(args)
    for image in os.listdir(IMAGEDIR):
        if "dummy_image" in image:
            continue
        print(f"===== generating mask for {image} =====")
        input_image = Image.open(os.path.join(IMAGEDIR, image))
        resolution = input_image.size
        out = Image.fromarray((model({"x": input_image})["pred_0"]["cca"]["mask"])*255)
        out.save(os.path.join(MASKDIR, os.path.basename(image)))
    for mask in os.listdir(MASKDIR):
        print(f"===== rescaling mask: {mask} =====")
        _mask = cv2.imread(os.path.join(MASKDIR, mask))
        rescaled_mask = cv2.resize(_mask, dsize=resolution, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(IMAGEDIR, f"dynamic_mask_{mask.split('.')[0]}.png"), rescaled_mask)


    os.system(f"python scripts/run.py  --scene workspace/ --save_snapshot /home/sersandr/synthetic_data/Pipeline/out/out_INSTANTNGP/{MODELNAME}.ingp")

def convert(args):
    AABB_SCALE = int(args.aabb_scale)
    IMAGEPATH = args.image_path
    TEXTPATH = args.text_path
    OUTPATH = args.out_path
    SKIPEARLY = args.skip_early

    try: 
        open(OUTPATH, "a").close()
    except Exception as e:
        print(f"Could not save transforms JSON to {OUTPATH}: {e}")
        sys.exit(1)

    print(f"outputting to {OUTPATH}...")
    cameras = {}
    with open(os.path.join(TEXTPATH, "cameras.txt"), "r") as f:
        camera_angle_x = math.pi / 2
        for line in f:
            if line[0] == '#':
                continue
            els = line.split(" ")
            camera = {}
            camera_id = int(els[0])
            camera["w"] = float(els[2])
            camera["h"] = float(els[3])
            camera["fl_x"] = float(els[4])
            camera["fl_y"] = float(els[4])
            camera["k1"] = 0
            camera["k2"] = 0
            camera["k3"] = 0
            camera["k4"] = 0
            camera["p1"] = 0
            camera["p2"] = 0
            camera["cx"] = camera["w"] / 2
            camera["cy"] = camera["h"] / 2
            camera["is_fisheye"] = False
            if els[1] == "SIMPLE_PINHOLE":
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
            elif els[1] == "PINHOLE":
                camera["fl_y"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
            elif els[1] == "SIMPLE_RADIAL":
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
            elif els[1] == "RADIAL":
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
                camera["k2"] = float(els[8])
            elif els[1] == "OPENCV":
                camera["fl_y"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
                camera["k1"] = float(els[8])
                camera["k2"] = float(els[9])
                camera["p1"] = float(els[10])
                camera["p2"] = float(els[11])
            elif els[1] == "SIMPLE_RADIAL_FISHEYE":
                camera["is_fisheye"] = True
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
            elif els[1] == "RADIAL_FISHEYE":
                camera["is_fisheye"] = True
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
                camera["k2"] = float(els[8])
            elif els[1] == "OPENCV_FISHEYE":
                camera["is_fisheye"] = True
                camera["fl_y"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
                camera["k1"] = float(els[8])
                camera["k2"] = float(els[9])
                camera["k3"] = float(els[10])
                camera["k4"] = float(els[11])
            else:
                print("Unknown camera model ", els[1])
			# fl = 0.5 * w / tan(0.5 * angle_x);
            camera["camera_angle_x"] = math.atan(camera["w"] / (camera["fl_x"] * 2)) * 2
            camera["camera_angle_y"] = math.atan(camera["h"] / (camera["fl_y"] * 2)) * 2
            camera["fovx"] = camera["camera_angle_x"] * 180 / math.pi
            camera["fovy"] = camera["camera_angle_y"] * 180 / math.pi

            print(f"camera {camera_id}:\n\tres={camera['w'],camera['h']}\n\tcenter={camera['cx'],camera['cy']}\n\tfocal={camera['fl_x'],camera['fl_y']}\n\tfov={camera['fovx'],camera['fovy']}\n\tk={camera['k1'],camera['k2']} p={camera['p1'],camera['p2']} ")
            cameras[camera_id] = camera

    if len(cameras) == 0:
        print("No cameras found!")
        sys.exit(1)

    with open(os.path.join(TEXTPATH,"images.txt"), "r") as f:
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        if len(cameras) == 1:
            camera = cameras[camera_id]
            out = {
				"camera_angle_x": camera["camera_angle_x"],
				"camera_angle_y": camera["camera_angle_y"],
				"fl_x": camera["fl_x"],
				"fl_y": camera["fl_y"],
				"k1": camera["k1"],
				"k2": camera["k2"],
				"k3": camera["k3"],
				"k4": camera["k4"],
				"p1": camera["p1"],
				"p2": camera["p2"],
				"is_fisheye": camera["is_fisheye"],
				"cx": camera["cx"],
				"cy": camera["cy"],
				"w": camera["w"],
				"h": camera["h"],
				"aabb_scale": AABB_SCALE,
				"frames": [],
			}
        else:
            out = {
				"frames": [],
				"aabb_scale": AABB_SCALE
			}

        up = np.zeros(3)
        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            if i < SKIPEARLY*2:
                continue
            if  i % 2 == 1:
                elems=line.split(" ") # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
				#name = str(PurePosixPath(Path(IMAGE_FOLDER, elems[9])))
				# why is this requireing a relitive path while using ^
                image_rel = os.path.relpath(IMAGEPATH)
                full_name = str(f"./{image_rel}/{'_'.join(elems[9:])}")
                name = '_'.join(elems[9:])
                shutil.copyfile(os.path.join(image_rel, name), os.path.join("workspace/images", name))
                b = sharpness(full_name)
                print(name, "sharpness=",b)
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3,1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                c2w = np.linalg.inv(m)
                if not args.keep_colmap_coords:
                    c2w[0:3,2] *= -1 # flip the y and z axis
                    c2w[0:3,1] *= -1
                    c2w = c2w[[1,0,2,3],:]
                    c2w[2,:] *= -1 # flip whole world upside down

                    up += c2w[0:3,1]

                frame = {"file_path":os.path.join("images", name),"sharpness":b,"transform_matrix": c2w}
                if len(cameras) != 1:
                    frame.update(cameras[int(elems[8])])
                out["frames"].append(frame)


    if args.keep_colmap_coords:
        flip_mat = np.array([
			[1, 0, 0, 0],
			[0, -1, 0, 0],
			[0, 0, -1, 0],
			[0, 0, 0, 1]
		])

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat) # flip cameras (it just works)
    else:
		# don't keep colmap coords - reorient the scene to be easier to work with

        up = up / np.linalg.norm(up)
        print("up vector was", up)
        R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
        R = np.pad(R,[0,1])
        R[-1, -1] = 1

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

		# find a central point they are all looking at
        print("computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in out["frames"]:
            mf = f["transform_matrix"][0:3,:]
            for g in out["frames"]:
                mg = g["transform_matrix"][0:3,:]
                p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
                if w > 0.00001:
                    totp += p*w
                    totw += w
        if totw > 0.0:
            totp /= totw
        print(totp) # the cameras are looking at totp

        if args.add_poses:
            ## generate dummy image
            size = Image.open(os.path.join("workspace/images", os.listdir("workspace/images")[0])).size

            dummy_img = Image.new("RGBA", (2, 2), (255, 0, 0, 0))
            draw = ImageDraw.Draw(dummy_img)

            dummy_img.save("workspace/images/dummy_image.png", "png", tansparency=100)

            ## generate additional poses for data synthesis
            print("generating additional poses for data synthesis")
            for _ in range(args.image_count):
                x = np.random.uniform(-5, 5)
                y = np.random.uniform(-5, 5)
                z = np.random.uniform(0, 5)

                c2w = elu_to_c2w([x, y, z], totp, [0.0, 0.0, 1.0])
                frame = {"file_path":os.path.join("images", "dummy_image.png"),"sharpness":100,"transform_matrix": c2w}
                out["frames"].append(frame)

                
        nframes = len(out["frames"])

        for f in out["frames"]:
            f["transform_matrix"][0:3,3] -= totp

        avglen = 0.
        for f in out["frames"]:
            avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
        avglen /= nframes
        print("avg camera distance from origin", avglen)
        for f in out["frames"]:
            f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
    print(nframes,"frames")
    print(f"writing {OUTPATH}")
    with open(OUTPATH, "w") as outfile:
        json.dump(out, outfile, indent=2)
	

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm

def qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ], [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ], [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
        ]
    ])

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
	# handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0: 
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom

def elu_to_c2w(eye, lookat, up):

    if isinstance(eye, list):
        eye = np.array(eye)
    if isinstance(lookat, list):
        lookat = np.array(lookat)
    if isinstance(up, list):
        up = np.array(up)

    l = eye - lookat
    if np.linalg.norm(l) < 1e-8:
        l[-1] = 1
    l = l / np.linalg.norm(l)

    s = np.cross(l, up)
    if np.linalg.norm(s) < 1e-8:
        s[0] = 1
    s = s / np.linalg.norm(s)
    uu = np.cross(s, l)

    rot = np.eye(3)
    rot[0, :] = -s
    rot[1, :] = uu
    rot[2, :] = l
    
    c2w = np.eye(4)
    c2w[:3, :3] = rot.T
    c2w[:3, 3] = eye

    return c2w


if __name__ == "__main__":
    args = parse_args()
    run_instantngp(args)

