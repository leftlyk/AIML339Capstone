import os
import json
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import scipy.io
import numpy as np


root = Path(os.getcwd())  # Fix: use current working directory
sys.path.append(str(root))

dataset_name = 'mpii'
dataset_path = os.path.join(root, dataset_name)

print(dataset_path)

# Format of filenames = [[mpii_img_1, mpii_img_2, ... (mpii_img_k)]]
filenames_file = os.path.join(dataset_path, f'{dataset_name}_filenames.txt')
with open(filenames_file, 'r') as f:
    filenames = f.read().split()

mpii_idx_to_jnt = {0: 'rankl', 1: 'rknee', 2: 'rhip', 5: 'lankl', 4: 'lknee', 3: 'lhip',
                   6: 'pelvis', 7: 'thorax', 8: 'upper_neck', 11: 'relb', 10: 'rwri', 9: 'head',
                   12: 'rsho', 13: 'lsho', 14: 'lelb', 15: 'lwri'}

# Load the mat file.
matlab_mpii = scipy.io.loadmat(os.path.join(dataset_path, 'mpii_annotations.mat'), struct_as_record=False)['RELEASE'][0, 0]

def crop_person(image, joints, margin=40):
    # Get all visible joint coordinates
    coords = np.array([[j[0], j[1]] for j in joints if j[2] > 0])
    if len(coords) == 0:
        return None, None
    x_min, y_min = coords.min(axis=0) - margin
    x_max, y_max = coords.max(axis=0) + margin
    x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
    x_max, y_max = min(image.shape[1], int(x_max)), min(image.shape[0], int(y_max))
    cropped = image[y_min:y_max, x_min:x_max]
    # Adjust joints to new crop
    new_joints = []
    for j in joints:
        if j[2] > 0:
            new_joints.append([j[0]-x_min, j[1]-y_min, j[2]])
    return cropped, new_joints

def save_single_person_crops_and_annotations(matlab_mpii, dataset_path, idx_to_jnt, out_dir='cropped_persons', ann_dir='annotations', ann_file='cropped_persons_annotations.json', max_images=1000):

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    annolist = matlab_mpii.__dict__['annolist'][0]
    single_person = matlab_mpii.__dict__['single_person']
    annotations = []
    saved = 0

    for img_idx in tqdm(range(len(annolist))):
        if saved >= max_images:
            break
        annotation = annolist[img_idx]
        img_name = annotation.__dict__['image'][0, 0].__dict__['name'][0]
        img_path = os.path.join(dataset_path, 'images', img_name)
        if not os.path.exists(img_path):
            #print(f"Image not found: {img_path}")
            continue
        image = plt.imread(img_path)
        person_ids = single_person[img_idx][0].flatten()
        for pidx, person in enumerate(person_ids - 1):
            try:
                annopoints = annotation.__dict__['annorect'][0, person].__dict__['annopoints'][0, 0]
                points = annopoints.__dict__['point'][0]
                # Prepare a dict for all joints, defaulting to not visible
                all_joints = {k: {'name': k, 'x': 0.0, 'y': 0.0, 'visible': 0, 'mpii_id': idx} for idx, k in idx_to_jnt.items()}
                for i in range(points.shape[0]):
                    x = points[i].__dict__['x'].flatten()[0]
                    y = points[i].__dict__['y'].flatten()[0]
                    id_ = points[i].__dict__['id'][0][0]
                    vis = points[i].__dict__['is_visible'].flatten()
                    vis = vis.item() if vis.size > 0 else 1
                    jnt_name = idx_to_jnt.get(id_, 'unknown')
                    if jnt_name in all_joints:
                        all_joints[jnt_name] = {
                            'name': jnt_name,
                            'x': float(x),
                            'y': float(y),
                            'visible': int(vis),
                            'mpii_id': int(id_)
                        }
                # Now build the list in the correct order
                ordered_joints = [all_joints[idx_to_jnt[i]] for i in sorted(idx_to_jnt.keys())]
                joints_for_crop = [[j['x'], j['y'], j['visible']] for j in ordered_joints]
                cropped_img, cropped_joints = crop_person(image, joints_for_crop)
                if cropped_img is None:
                    #print(f"No visible joints for {img_name} person {pidx+1}")
                    continue

                cropped_idx = 0
                for j in ordered_joints:
                    if j['visible'] > 0:
                        j['x'] = float(cropped_joints[cropped_idx][0])
                        j['y'] = float(cropped_joints[cropped_idx][1])
                        cropped_idx += 1

                # Set to (0,0) if not visible or out of bounds
                height, width = cropped_img.shape[:2]
                for j in ordered_joints:
                    if (j['visible'] == 0 or
                        j['x'] < 0 or j['y'] < 0 or
                        j['x'] >= width or j['y'] >= height):
                        j['x'] = 0.0
                        j['y'] = 0.0

                # Save cropped image
                out_img_name = f"{os.path.splitext(img_name)[0]}_person{pidx+1}.jpg"
                out_img_path = os.path.join(out_dir, out_img_name)
                plt.imsave(out_img_path, cropped_img)
                print(f"Saved: {out_img_path}")
                # Prepare annotation entry
                annotation_entry = {
                    "image": out_img_name,
                    "original_image": img_name,
                    "person_index": int(pidx+1),
                    "joints": ordered_joints
                }
                annotations.append(annotation_entry)
                saved += 1
                if saved >= max_images:
                    break
            except Exception as e:
                #print(f"Exception for {img_name} person {pidx+1}: {e}")
                continue

    # Save all annotations to a JSON file
    ann_file = os.path.join(ann_dir, ann_file)
    with open(ann_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    #print(f"Saved {saved} cropped images and annotations to {out_dir} and {ann_file}")

# Usage:
save_single_person_crops_and_annotations(
    matlab_mpii, dataset_path, mpii_idx_to_jnt,
    out_dir='cropped_persons',
    ann_dir='annotations',
    ann_file='cropped_persons_annotations.json',
    max_images=40000
)