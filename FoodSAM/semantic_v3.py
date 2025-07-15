import sys
# sys.path.append('.')
# sys.path.append('./SAM')
# sys.path.append('./mmseg')
import argparse
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from mmcv.utils import DictAction
import argparse
import json
import os
import numpy as np
from typing import Any, Dict, List
import shutil, logging
from FoodSAM_tools.predict_semantic_mask_v2 import semantic_predict
from FoodSAM_tools.enhance_semantic_masks_v3 import enhance_masks
from FoodSAM_tools.evaluate_foodseg103 import evaluate

parser = argparse.ArgumentParser(
    description=(
        "Runs SAM automatic mask generation and semantic segmentation on an input image or directory of images, "
        "and then enhance the semantic masks based on SAM output masks"
    )
)
parser.add_argument(
    "--data_root",
    type=str,
    default='dataset/FoodSeg103/Images',
    help="Path to folder of images and masks.",
)
parser.add_argument(
    "--img_dir",
    type=str,
    default='img_dir/test',
    help="dir name of images",
)
parser.add_argument(
    "--ann_dir",
    type=str,
    default='ann_dir/test',
    help="dir name of gt masks.",
)

parser.add_argument(
    "--img_path",
    type=str,
    default=None,
    help="dir name of imgs.",
)
parser.add_argument(
    "--output",
    type=str,
    default='Output/Semantic_Results',
    help=(
        "Path to the directory where results will be output. Output will be a folder "
    ),
)
parser.add_argument(
    "--SAM_checkpoint",
    type=str,
    default="ckpts/sam_vit_h_4b8939.pth",
    help="The path to the SAM checkpoint to use for mask generation.",
)
parser.add_argument('--semantic_config', default="configs/SETR_MLA_768x768_80k_base.py", help='test config file path of mmseg')
parser.add_argument('--semantic_checkpoint', default="ckpts/SETR_MLA/iter_80000.pth", help='checkpoint file of mmseg')
parser.add_argument(
    "--model-type",
    type=str,
    default='vit_h',
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")


parser.add_argument(
    '--aug-test', action='store_true', help='Use Flip and Multi scale aug')

parser.add_argument(
    '--options', nargs='+', action=DictAction, help='custom options')

parser.add_argument(
    '--eval-options',
    nargs='+',
    action=DictAction,
    help='custom options for evaluation'
)

parser.add_argument(
    '--color_list_path',
    type=str,
    default="FoodSAM/FoodSAM_tools/color_list_v2.npy",
    help='the color used to draw for each label'
)

parser.add_argument(
    "--category_txt",
    default="FoodSAM/FoodSAM_tools/category_id_files/foodseg103_category_id.txt",
    help='the category name of each label'
)
parser.add_argument(
    "--num_class",
    default=104,
    help='the total number of classes including background'
)
parser.add_argument(
    "--area_thr",
    default=0,
    help='the area threshold used to enhance masks'
)
parser.add_argument(
    "--ratio_thr",
    default=0.5,
    help='the ratio threshold used to enhance masks'
)
parser.add_argument(
    "--top_k",
    default=80,
    help='only the top k SAM masks sorted by SAM will be kept. '
)

parser.add_argument(
    "--eval",
    action='store_true',
    help='evaluate the semantic results'
)

parser.add_argument(
    "--email",
    default='xienli1025@gmail.com',
    help='user login email'
)

parser.add_argument(
    "--type",
    default=2,
    help='type of kernel in sharpen image'
)

parser.add_argument(
    "--h",
    default=12,
    help='hyperparam of denosing image'
)

parser.add_argument(
    "--clipLimit",
    default=1.2,
    help='hyperparam of enhence contrast'
)

parser.add_argument(
    "--s",
    default=8,
    help='hyperparam of enhence contrast'
)

parser.add_argument(
    "--blur_intensity",
    default=21,
    help='hyperparam of gaussian blurring'
)

parser.add_argument(
    "--focus_size",
    default=0.7,
    help='hyperparam of gaussian blurring'
)

amg_settings = parser.add_argument_group("AMG Settings")
amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    os.makedirs(os.path.join(path, "sam_mask"), exist_ok=True)
    masks_array = []
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        masks_array.append(mask.copy())
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, "sam_mask" ,filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)

    masks_array = np.stack(masks_array, axis=0)
    np.save(os.path.join(path, "sam_mask" ,"masks.npy"), masks_array)
    metadata_path = os.path.join(path, "sam_metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))
    return


def create_logger(save_folder):
    
    log_file = f"sam_process.log"
    final_log_file = os.path.join(save_folder, log_file)

    logging.basicConfig(
        format=
        '[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(final_log_file, mode='w'),
            logging.StreamHandler()
        ])                        
    logger = logging.getLogger()
    print(f"Create Logger success in {final_log_file}")
    return logger

def normalize_for_inference(image):
    return image / 255.0  # Adjust if the model requires a different normalization

def convert_color_space(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def denoise_image(image, h):
    # h:8-15
    h = float(h)
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)

def enhance_contrast(image,clipLimit,s):
    clipLimit = float(clipLimit)
    s = int(s)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(s, s))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def convert_to_uint8(image):
    # If the image is in float64, convert it to uint8
    if image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)  # Assuming the image was normalized to [0, 1]
    return image

def sharpen_image(image, type):
    type = int(type)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    gaussian_kernel = np.array([[1, 2, 1], 
                                [2, 4, 2], 
                                [1, 2, 1]]) / 16
    gaussian_kernel_v2 = np.array([ [1, 4, 6, 4, 1],
                                    [4, 16, 24, 16, 4],
                                    [6, 24, 36, 24, 6],
                                    [4, 16, 24, 16, 4],
                                    [1, 4, 6, 4, 1]]) / 256


    if type == 1:
        return cv2.filter2D(image, -1, kernel)
    elif type == 2:
        return cv2.filter2D(image, -1, gaussian_kernel)
    else:
        return cv2.filter2D(image, -1, gaussian_kernel_v2)


def apply_gaussian_blur_focus(image, blur_intensity=21, focus_size=0.7):
    blur_intensity = int(blur_intensity)
    focus_size = float(focus_size)
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Create a blurred version of the image
    blurred_image = cv2.GaussianBlur(image, (blur_intensity, blur_intensity), 0)
    
    # Create a mask with a circular focus region (sharp) and smooth blurred edges
    mask = np.zeros((h, w), dtype=np.uint8)
    focus_radius = int(min(h, w) * focus_size / 2)
    
    # Draw a filled white circle at the center (which will remain sharp)
    center = (w // 2, h // 2)
    cv2.circle(mask, center, focus_radius, (255), -1)
    
    # Create the blurred image by blending sharp and blurred images using the mask
    mask = cv2.GaussianBlur(mask, (blur_intensity, blur_intensity), 0)
    mask = mask.astype(np.float32) / 255.0  # Normalize mask to [0, 1]
    
    # Combine sharp and blurred images using the mask
    result = (image * mask[..., np.newaxis] + blurred_image * (1 - mask[..., np.newaxis])).astype(np.uint8)
    
    return result

def preprocess_for_inference(image, type, h, clipLimit, s, blur_intensity, focus_size):
    image = convert_color_space(image)
    image = enhance_contrast(image, clipLimit, s)
    image = sharpen_image(image, type)
    image = denoise_image(image, h)
    image = apply_gaussian_blur_focus(image, blur_intensity, focus_size)
    image = convert_to_uint8(image)
    return image

def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output, exist_ok=True)
    logger = create_logger(args.output)
    logger.info("running sam!")
    sam = sam_model_registry[args.model_type](checkpoint=args.SAM_checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)
    
    assert args.data_root or args.img_path
    if args.img_path:
        targets = [args.img_path]
    else:
        img_folder = os.path.join(args.data_root, args.img_dir)
        targets = [
            f for f in os.listdir(img_folder) if not os.path.isdir(os.path.join(img_folder, f))
        ]
        targets = [os.path.join(img_folder, f) for f in targets]

    for t in targets:
        logger.info(f"Processing '{t}'...")
        image = cv2.imread(t)
        if image is None:
            logger.error(f"Could not load '{t}' as an image, skipping...")
            continue
        
        image = preprocess_for_inference(image, args.type, args.h, args.clipLimit, args.s, args.blur_intensity, args.focus_size)
        masks = generator.generate(image)
        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, base)
        os.makedirs(save_base, exist_ok=True)
        cv2.imwrite(os.path.join(save_base, "preprocessed.jpg"), image)
        write_masks_to_folder(masks, save_base)
        shutil.copyfile(t, os.path.join(save_base, "input.jpg"))
    logger.info("sam done!\n")

    
    logger.info("running semantic seg model!")
    semantic_predict(args.data_root, args.img_dir, args.ann_dir, args.semantic_config, args.options, args.aug_test, args.semantic_checkpoint, args.eval_options, args.output, args.color_list_path, args.email, args.img_path)
    logger.info("semantic predict done!\n")
    

    logger.info("Gabriel enhance semantic masks")
    enhance_masks(args.output, args.category_txt, args.color_list_path, args.img_path, num_class=args.num_class, area_thr=args.area_thr, ratio_thr=args.ratio_thr, top_k=args.top_k, email=args.email)
    logger.info("done!\n")

    

    if args.eval and not args.img_path:
        ann_folder = os.path.join(args.data_root, args.ann_dir)
        evaluate(args.output, ann_folder, args.num_class)

    #if args.eval:
    #    ann_folder = os.path.join(args.data_root, args.ann_dir)
    #    evaluate(args.output, ann_folder, args.num_class)

    logger.info("The results saved in {}!\n".format(args.output))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
