# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import glob
import gzip
import json
import os
from pathlib import Path
import sys
from typing import Any
import cv2
import distinctipy
import imageio
from natsort import natsorted
import pickle
from tqdm import trange

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision

import supervision as sv
from supervision.draw.color import Color, ColorPalette
import dataclasses

import open_clip

# import Grounded SAM
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# This is needed for the following imports in this file
sys.path.append(os.environ["GSA_PATH"]) 
sys.path.append(os.environ["EFFICIENTSAM_PATH"])

import torchvision.transforms as TS

GROUNDING_DINO_CONFIG_PATH = os.environ["GROUNDING_DINO_CONFIG_PATH"]
GROUNDING_DINO_CHECKPOINT_PATH = os.environ["GROUNDING_DINO_CHECKPOINT_PATH"]

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = os.environ["SAM_ENCODER_VERSION"]
SAM_CHECKPOINT_PATH = os.environ["SAM_CHECKPOINT_PATH"]

# Tag2Text checkpoint
TAG2TEXT_CHECKPOINT_PATH = os.environ["TAG2TEXT_CHECKPOINT_PATH"]
RAM_CHECKPOINT_PATH = os.environ["RAM_CHECKPOINT_PATH"]

class Dataset():
    def __init__(self, args) -> None:
        self.input_folder = args.input_folder
        assert self.input_folder.exists(), f"Input folder {self.input_folder} does not exist. "

        self.detection_save_folder = self.input_folder / f"gsa_det_{args.class_set}_{args.sam_variant}"
        self.detection_save_folder.mkdir(exist_ok=True)

        self.vis_save_folder = self.input_folder / f"gsa_vis_{args.class_set}_{args.sam_variant}"
        self.vis_save_folder.mkdir(exist_ok=True)

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

class AriaDataset(Dataset):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

        transform_path = self.input_folder / "transforms.json"
        with open(transform_path) as json_file:
            frames = json.loads(json_file.read())["frames"]
        
        # Only keep the RGB images
        self.frames = [f for f in frames if f['camera_name'] == 'rgb']

        self.frames.sort(key=lambda f: f["image_path"])

    def __getitem__(self, index: int) -> Any:
        subpath = self.frames[index]["image_path"]
        image_path = self.input_folder / subpath
        image_filename = subpath[:-4] # remove the .png/.jpg extension

        return image_path, image_filename
    
    def __len__(self) -> int:
        return len(self.frames)
    
class ReplicaDataset(Dataset):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.rgb_paths = natsorted(glob.glob(str(args.input_folder / "rgb" / "rgb_*.png")))

    def __getitem__(self, index: int) -> Any:
        image_path = self.rgb_paths[index]
        image_filename = image_path.split('/')[-1].split('.')[0]
        return image_path, image_filename

    def __len__(self) -> int:
        return len(self.rgb_paths)
    
class ColmapDataset(Dataset):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.rgb_paths = natsorted(glob.glob(str(args.input_folder / "images" / "*.JPG")))

    def __getitem__(self, index: int) -> Any:
        image_path = self.rgb_paths[index]
        image_filename = image_path.split('/')[-1].split('.')[0]
        return image_path, image_filename
    
    def __len__(self) -> int:
        return len(self.rgb_paths)
    
class BlenderDataset(Dataset):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.rgb_paths = []
        self.filenames = []
        for subset in ["train", "val", "test"]:
            subset_folder = self.input_folder / subset
            for image_path in natsorted(glob.glob(str(subset_folder / "*.png"))):
                self.rgb_paths.append(image_path)
                filename = image_path.split('/')[-1].split('.')[0]
                subpath = image_path.split('/')[-2] + '/' + filename
                self.filenames.append(subpath)

    def __getitem__(self, index: int) -> Any:
        image_path = self.rgb_paths[index]
        image_filename = self.filenames[index]
        return image_path, image_filename

    def __len__(self) -> int:
        return len(self.rgb_paths)
    
class NerfieDataset(Dataset):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.rgb_paths = natsorted(glob.glob(str(args.input_folder / "rgb" / "1x" / "*.png")))

    def __getitem__(self, index: int) -> Any:
        image_path = self.rgb_paths[index]
        image_filename = image_path.split('/')[-1].split('.')[0]
        return image_path, image_filename
    
    def __len__(self) -> int:
        return len(self.rgb_paths)
        
        
def compute_clip_features(image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device):
    backup_image = image.copy()
    
    image = Image.fromarray(image)
    
    # padding = args.clip_padding  # Adjust the padding amount as needed
    padding = 20  # Adjust the padding amount as needed
    
    image_crops = []
    image_feats = []
    text_feats = []
    
    for idx in range(len(detections.xyxy)):
        # Get the crop of the mask with padding
        x_min, y_min, x_max, y_max = detections.xyxy[idx]

        # Check and adjust padding to avoid going beyond the image borders
        image_width, image_height = image.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        # Apply the adjusted padding
        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        
        # Get the preprocessed image for clip from the crop 
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0).to("cuda")

        crop_feat = clip_model.encode_image(preprocessed_image)
        crop_feat /= crop_feat.norm(dim=-1, keepdim=True)
        
        class_id = detections.class_id[idx]
        tokenized_text = clip_tokenizer([classes[class_id]]).to("cuda")
        text_feat = clip_model.encode_text(tokenized_text)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        
        crop_feat = crop_feat.cpu().numpy()
        text_feat = text_feat.cpu().numpy()

        image_crops.append(cropped_image)
        image_feats.append(crop_feat)
        text_feats.append(text_feat)
        
    # turn the list of feats into np matrices
    image_feats = np.concatenate(image_feats, axis=0)
    text_feats = np.concatenate(text_feats, axis=0)

    return image_crops, image_feats, text_feats


def vis_result_fast(
    image: np.ndarray, 
    detections: sv.Detections, 
    classes: list[str], 
    color: Color | ColorPalette = ColorPalette.default(), 
    instance_random_color: bool = False,
    draw_bbox: bool = True,
) -> np.ndarray:
    '''
    Annotate the image with the detection results. 
    This is fast but of the same resolution of the input image, thus can be blurry. 
    '''
    # annotate image with detections
    box_annotator = sv.BoundingBoxAnnotator(
        color = color,
    )
    label_annontator = sv.LabelAnnotator(
        text_scale=0.3,
        text_thickness=1,
        text_padding=2,
    )
    mask_annotator = sv.MaskAnnotator(
        color = color,
        opacity=0.35,
    )
    labels = [
        f"{classes[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _, _
        in detections]
    
    if instance_random_color:
        # generate random colors for each segmentation
        # First create a shallow copy of the input detections
        detections = dataclasses.replace(detections)
        detections.class_id = np.arange(len(detections))
        
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    
    if draw_bbox:
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annontator.annotate(scene=annotated_image, detections=detections, labels=labels)
    return annotated_image, labels

def get_sam_mask_generator(variant:str, device: str | int) -> SamAutomaticMaskGenerator:
    if variant == "sam":
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device)
        # mask_generator = SamAutomaticMaskGenerator(
        #     model=sam,
        #     points_per_side=12,
        #     points_per_batch=144,
        #     pred_iou_thresh=0.88,
        #     stability_score_thresh=0.95,
        #     crop_n_layers=0,
        #     min_mask_region_area=100,
        # )
        mask_generator = SamAutomaticMaskGenerator(sam) # Use the default arguments
        return mask_generator
    elif variant == "fastsam":
        raise NotImplementedError
        # from ultralytics import YOLO
        # from FastSAM.tools import *
        # FASTSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/FastSAM-x.pt")
        # model = YOLO(args.model_path)
        # return model
    else:
        raise NotImplementedError

def get_sam_predictor(variant: str, device: str | int) -> SamPredictor:
    if variant == "sam":
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device)
        sam_predictor = SamPredictor(sam)
        return sam_predictor
        
    # TODO: (low priority) add support for other variants, as shown below
    # elif variant == "mobilesam":
    #     from MobileSAM.setup_mobile_sam import setup_model
    #     MOBILE_SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/mobile_sam.pt")
    #     checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
    #     mobile_sam = setup_model()
    #     mobile_sam.load_state_dict(checkpoint, strict=True)
    #     mobile_sam.to(device=device)
        
    #     sam_predictor = SamPredictor(mobile_sam)
    #     return sam_predictor

    # elif variant == "lighthqsam":
    #     from LightHQSAM.setup_light_hqsam import setup_model
    #     HQSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/sam_hq_vit_tiny.pth")
    #     checkpoint = torch.load(HQSAM_CHECKPOINT_PATH)
    #     light_hqsam = setup_model()
    #     light_hqsam.load_state_dict(checkpoint, strict=True)
    #     light_hqsam.to(device=device)
        
    #     sam_predictor = SamPredictor(light_hqsam)
    #     return sam_predictor
        
    elif variant == "fastsam":
        raise NotImplementedError
    else:
        raise NotImplementedError
    
# Prompting SAM with detected boxes
def get_sam_segmentation_from_xyxy(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

# The SAM based on automatic mask generation, without bbox prompting
def get_sam_segmentation_dense(
    variant:str, model: Any, image: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    The SAM based on automatic mask generation, without bbox prompting
    
    Args:
        model: The mask generator or the YOLO model
        image: )H, W, 3), in RGB color space, in range [0, 255]
        
    Returns:
        mask: (N, H, W)
        xyxy: (N, 4)
        conf: (N,)
    '''
    if variant == "sam":
        results = model.generate(image)
        mask = []
        xyxy = []
        conf = []
        for r in results:
            mask.append(r["segmentation"])
            r_xyxy = r["bbox"].copy()
            # Convert from xyhw format to xyxy format
            r_xyxy[2] += r_xyxy[0]
            r_xyxy[3] += r_xyxy[1]
            xyxy.append(r_xyxy)
            conf.append(r["predicted_iou"])
        mask = np.array(mask)
        xyxy = np.array(xyxy)
        conf = np.array(conf)
        return mask, xyxy, conf
    elif variant == "fastsam":
        # The arguments are directly copied from the GSA repo
        results = model(
            image,
            imgsz=1024,
            device="cuda",
            retina_masks=True,
            iou=0.9,
            conf=0.4,
            max_det=100,
        )
        raise NotImplementedError
    else:
        raise NotImplementedError
    
def process_tag_classes(text_prompt:str, add_classes:list[str]=[], remove_classes:list[str]=[]) -> list[str]:
    '''
    Convert a text prompt from Tag2Text to a list of classes. 
    '''
    classes = text_prompt.split(',')
    classes = [obj_class.strip() for obj_class in classes]
    classes = [obj_class for obj_class in classes if obj_class != '']
    
    for c in add_classes:
        if c not in classes:
            classes.append(c)
    
    for c in remove_classes:
        classes = [obj_class for obj_class in classes if c not in obj_class.lower()]
    
    return classes



@torch.no_grad()
def main(args: argparse.Namespace):
    ### Initialize the SAM model ###
    if args.class_set == "none":
        # Generate the masks in dense fashion
        mask_generator = get_sam_mask_generator(args.sam_variant, args.device)
    else:
        ### Initialize the Grounding DINO model ###
        grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, 
            device=args.device
        )
        # Generate the masks by bounding boxes
        sam_predictor = get_sam_predictor(args.sam_variant, args.device)
        
    ###
    # Initialize the CLIP model
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_model = clip_model.to(args.device)
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    ### Initialize the tagging model, or get the class set to detect ###
    if args.class_set in ["tag2text", "ram"]:
        if args.class_set == "tag2text":
            # The class set will be computed by tag2text on each image
            # filter out attributes and action categories which are difficult to grounding
            delete_tag_index = []
            for i in range(3012, 3429):
                delete_tag_index.append(i)

            specified_tags='None'
            # load model
            tagging_model = tag2text.tag2text_caption(pretrained=TAG2TEXT_CHECKPOINT_PATH,
                                                    image_size=384,
                                                    vit='swin_b',
                                                    delete_tag_index=delete_tag_index)
            # threshold for tagging
            # we reduce the threshold to obtain more tags
            tagging_model.threshold = 0.64 
        elif args.class_set == "ram":
            tagging_model = tag2text.ram(pretrained=RAM_CHECKPOINT_PATH,
                                         image_size=384,
                                         vit='swin_l')
            
        tagging_model = tagging_model.eval().to(args.device)
        
        # initialize Tag2Text
        tagging_transform = TS.Compose([
            TS.Resize((384, 384)),
            TS.ToTensor(), 
            TS.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
        
        classes = None
    elif args.class_set == "none":
        classes = ['item']
    elif args.class_set == "input":
        classes = args.input_classes.split(',')
    else:
        raise ValueError("Unknown args.class_set: ", args.class_set)

    if args.class_set == "input":
        print("There are total", len(classes), "classes to detect. They are: ", classes)
    elif args.class_set == "none":
        print("Skipping tagging and detection models. ")
    else:
        print(f"{args.class_set} will be used to detect classes. ")

    ##### initialize the dataset #####
    rotate_back = False
    if (args.input_folder / "global_points.csv.gz").exists():
        print(f"Found global_points.csv.gz file, assuming Aria data set!")
        dataset = AriaDataset(args)
        rotate_back = True
    elif (args.input_folder / "traj_w_c.txt").exists():
        print(f"Fount traj_w_c.txt file. Assuming Replica Semantic dataset!")
        dataset = ReplicaDataset(args)
    elif (args.input_folder / "sparse").exists():
        print("Found sparse folder, assuming Colmap data set!")
        dataset = ColmapDataset(args)
    elif (args.input_folder / "transforms_train.json").exists():
        print("Found transforms_train.json file, assuming Blender data set!")
        dataset = BlenderDataset(args)
    elif (args.input_folder / "dataset.json").exists():
        print("Found dataset.json file, assuming Nerfies data set!")
        dataset = NerfieDataset(args)
    else:
        raise ValueError("Unknown dataset type. ")

    annotated_frames = []
    global_classes = []

    # frames = frames[:30]
    for idx in trange(0, len(dataset), args.stride):
        # image_path = args.input_folder / frames[idx]["image_path"]
        # image_filename = image_path.name.split('.')[0]
        image_path, image_filename = dataset[idx]

        image_pil = Image.open(image_path)
        # image_pil = image_pil.resize((args.output_width, args.output_height))
        longer_side = min(max(image_pil.size), args.max_longer_side)
        resize_scale = float(longer_side) / max(image_pil.size)
        image_pil = image_pil.resize(
            (int(image_pil.size[0] * resize_scale), int(image_pil.size[1] * resize_scale))
        )
        # If image is RGBA, drop the alpha channel
        if image_pil.mode == "RGBA":
            image_pil = image_pil.convert("RGB")
        
        if rotate_back:
            image_pil = image_pil.rotate(-90, expand=True)
        image_rgb = np.array(image_pil)
        image_bgr = image_rgb[:, :, ::-1].copy()

        ### Tag2Text ###
        if args.class_set in ["ram", "tag2text"]:
            raw_image = image_pil.resize((384, 384))
            raw_image = tagging_transform(raw_image).unsqueeze(0).to(args.device)
            
            if args.class_set == "ram":
                res = inference_ram.inference(raw_image , tagging_model)
                caption="NA"
            elif args.class_set == "tag2text":
                res = inference_tag2text.inference(raw_image , tagging_model, specified_tags)
                caption=res[2]

            # Currently ", " is better for detecting single tags
            # while ". " is a little worse in some case
            text_prompt=res[0].replace(' |', ',')
            
            # Add "other item" to capture objects not in the tag2text captions. 
            # Remove "xxx room", otherwise it will simply include the entire image
            # Also hide "wall" and "floor" for now...
            add_classes = ["other item"]
            remove_classes = [
                "room", "kitchen", "office", "house", "home", "building", "corner",
                "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium", 
                "apartment", "image", "city", "blue", "skylight", "hallway", 
                "bureau", "modern", "salon", "doorway", "wall lamp", "pantry"
            ]
            bg_classes = ["wall", "floor", "ceiling"]

            if args.add_bg_classes:
                add_classes += bg_classes
            else:
                remove_classes += bg_classes

            classes = process_tag_classes(
                text_prompt, 
                add_classes = add_classes,
                remove_classes = remove_classes,
            )

        # add classes to global classes
        for c in classes:
            if c not in global_classes:
                global_classes.append(c)
        
        if args.accumu_classes:
            # Use all the classes that have been seen so far
            classes = global_classes

        ### Detection and segmentation ###
        if args.class_set == "none":
            # Directly use SAM in dense sampling mode to get segmentation
            mask, xyxy, conf = get_sam_segmentation_dense(
                args.sam_variant, mask_generator, image_rgb)

            detections = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=np.zeros_like(conf).astype(int),
                mask=mask,
            )

            # Remove the bounding boxes that are too large (they tend to capture the entire image)
            areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (detections.xyxy[:, 3] - detections.xyxy[:, 1])
            area_ratios = areas / (image_rgb.shape[0] * image_rgb.shape[1])
            valid_idx = area_ratios < 0.6
            detections.xyxy = detections.xyxy[valid_idx]
            detections.confidence = detections.confidence[valid_idx]
            detections.class_id = detections.class_id[valid_idx]
            detections.mask = detections.mask[valid_idx]
        else:
            # Using GroundingDINO to detect and SAM to segment
            detections = grounding_dino_model.predict_with_classes(
                image=image_bgr, # This function expects a BGR image...
                classes=classes,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )
            
            if len(detections.class_id) > 0:
                ### Non-maximum suppression ###
                # print(f"Before NMS: {len(detections.xyxy)} boxes")
                nms_idx = torchvision.ops.nms(
                    torch.from_numpy(detections.xyxy), 
                    torch.from_numpy(detections.confidence), 
                    args.nms_threshold
                ).numpy().tolist()
                # print(f"After NMS: {len(detections.xyxy)} boxes")

                detections.xyxy = detections.xyxy[nms_idx]
                detections.confidence = detections.confidence[nms_idx]
                detections.class_id = detections.class_id[nms_idx]
                
                # Somehow some detections will have class_id=-1, remove them
                valid_idx = detections.class_id != -1
                detections.xyxy = detections.xyxy[valid_idx]
                detections.confidence = detections.confidence[valid_idx]
                detections.class_id = detections.class_id[valid_idx]

                # Remove the bounding boxes that are too large (they tend to capture the entire image)
                areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (detections.xyxy[:, 3] - detections.xyxy[:, 1])
                area_ratios = areas / (image_rgb.shape[0] * image_rgb.shape[1])
                valid_idx = area_ratios < 0.5
                detections.xyxy = detections.xyxy[valid_idx]
                detections.confidence = detections.confidence[valid_idx]
                detections.class_id = detections.class_id[valid_idx]
                
                ### Segment Anything ###
                detections.mask = get_sam_segmentation_from_xyxy(
                    sam_predictor=sam_predictor,
                    image=image_rgb,
                    xyxy=detections.xyxy
                )

        ### Compute CLIP features ###
        if not args.no_clip:
            image_crops, image_feats, text_feats = compute_clip_features(
                image_rgb, detections, clip_model, clip_preprocess, clip_tokenizer, classes, args.device)
        else:
            image_crops, image_feats, text_feats = None, None, None

        ### Save the detection results ###
        detection_save_path = dataset.detection_save_folder / f"{image_filename}.pkl.gz"
        detection_save_path.parent.mkdir(exist_ok=True, parents=True)
        det_results = {
            "image_path": image_path,
            "xyxy": detections.xyxy,
            "confidence": detections.confidence,
            "class_id": detections.class_id,
            "mask": detections.mask,
            "classes": classes,
            "image_crops": image_crops,
            "image_feats": image_feats,
            "text_feats": text_feats,
        }
        with gzip.open(str(detection_save_path), 'wb') as f:
            pickle.dump(det_results, f)

            
        ### Visualize results and save ###
        annotated_image, labels = vis_result_fast(
            image_rgb, detections, classes, 
            instance_random_color = args.class_set=="none",
            draw_bbox = args.class_set!="none",
        )

        vis_save_path = dataset.vis_save_folder / f"{image_filename}.png"
        vis_save_path.parent.mkdir(exist_ok=True, parents=True)
        imageio.imwrite(vis_save_path, annotated_image)
        
        # plt.figure(figsize=(10, 10))
        # plt.imshow(annotated_image)
        # plt.title(f"Frame {idx}")
        # plt.show()
        # cv2.imwrite(vis_save_path, annotated_image)
        annotated_frames.append(annotated_image)

    # Save the annotated frames as a video
    annotated_frames = np.stack(annotated_frames, axis=0)

    imageio.mimwrite(
        args.input_folder / f"gsa_vis_{args.class_set}_{args.sam_variant}.mp4",
        annotated_frames,
        fps=20,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", type=Path, default=Path("/home/qgu/local/data/aria_gs_v10/kitchen"),
                        help="Path to the input Aria sequence.")
    parser.add_argument("--stride", type=int, default=1,
                        help="The stride in subsampling a subset from all input frames. ")
    
    parser.add_argument("--sam_variant", type=str, default="sam", 
                        choices=["sam", "mobilesam", "lighthqsam", "fastsam"],
                        help="The variant of the SAM model to use.")
    
    parser.add_argument("--class_set", type=str, default="ram",
                        choices=["ram", 'tag2text', 'input', 'none'],
                        help="The class set to use for the SAM model")
    parser.add_argument("--input_classes", type=str, default="box,bottle,can,light", 
                        help="The text prompt for coloring the point cloud")
    
    parser.add_argument("--no_clip", action="store_true",
                        help="If set, do not compute CLIP features. ")
    
    parser.add_argument("--text_threshold", type=float, default=0.1, 
                        help="The text threshold for the Grounding DINO model. ")
    parser.add_argument("--box_threshold", type=float, default=0.1,
                        help="The box threshold for the Grounding DINO model. ")
    parser.add_argument("--nms_threshold", type=float, default=0.5,
                        help="The NMS threshold for the Grounding DINO model. ")
    
    parser.add_argument("--add_bg_classes", action="store_true", 
                        help="If set, add background classes (wall, floor, ceiling) to the class set. ")
    parser.add_argument("--accumu_classes", action="store_true",
                        help="if set, the class set will be accumulated over frames")
    
    parser.add_argument("--max_longer_side", type=int, default=640, 
                        help="The maximum longer side of the output masks. ")

    
    parser.add_argument("--device", type=str, default="cuda:0", 
                        help="The device to use for the inference. ")

    args = parser.parse_args()

    if args.class_set in ['ram', 'tag2text']:
        sys.path.append(os.environ["TAG2TEXT_PATH"])
        from Tag2Text.models import tag2text
        from Tag2Text import inference_tag2text, inference_ram
    
    main(args)
