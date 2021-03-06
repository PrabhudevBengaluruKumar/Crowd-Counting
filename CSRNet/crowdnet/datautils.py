import random
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from torch import Tensor
from torchvision import transforms


def load_dataset_paths(dataset_root: Path, img_extensions: set) -> List[Tuple[Path, Path]]:
    """ loads dataset paths

    Args:
        dataset_root (Path): path to the data
        img_extensions (set): set of image extensions

    Returns:
        List[Tuple[Path, Path]]: list of tuples of image paths and ground truth paths
    """
    images_path = dataset_root / "images"
    gt_path = dataset_root / "gt"

    img_paths = [p for p in images_path.iterdir() if p.suffix in img_extensions]
    gt_paths = [gt_path / f"{img_path.stem}.txt" for img_path in img_paths]
    return list(zip(img_paths, gt_paths))


def load_gt(path: Path) -> np.ndarray:
    """ loads ground truth from file

    Args:
        path (Path): path to the ground truth file
    Returns:
        np.ndarray: ground truth
    """
    with path.open("r") as f:
        gt = [list(map(int, line.rstrip().split())) for line in f]
    assert all([len(target) == 6 for target in gt]), f"Wrong target format {path}"
    return np.array(gt, dtype=np.int32)


def load_image(img_path: Union[Path, str]) -> Image.Image:
    """ loads image from file and converts it to RGB

    Args:
        img_path (Union[Path, str]): path to the image

    Returns:
        Image.Image: image
    """
    return Image.open(img_path).convert("RGB")


def draw_gt_labels(img: np.ndarray, gt_labels: np.ndarray) -> None:
    """ draws ground truth labels on the image

    Args:
        img (np.ndarray): image
        gt_labels (np.ndarray): ground truth labels

    Returns:
        _type_: image with ground truth labels
    """
    img = np.array(img)
    for gt_label in gt_labels:
        keypoint = gt_label[:2]
        xy_min = keypoint - gt_label[2:4]
        xy_max = keypoint + gt_label[2:4]

        img = cv2.circle(img, tuple(keypoint), 4, (255, 0, 0), -1)
        img = cv2.rectangle(img, tuple(xy_min), tuple(xy_max), (0, 255, 0))
    return img


def gauss2D(shape: Tuple[int, int], sigma: float = 0.5) -> np.ndarray:
    """ function to create a 2D gaussian kernel

    Args:
        shape (Tuple[int, int]): shape of the kernel
        sigma (float, optional): standard deviaiton . Defaults to 0.5.

    Returns:
        np.ndarray: 2D gaussian kernel
    """
    my, mx = [(x - 1) / 2 for x in shape]
    y, x = np.ogrid[-my:my + 1, -mx:mx + 1]
    gmap = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    normalizer = gmap.sum()
    if normalizer != 0:
        gmap /= normalizer
    return gmap


def generate_density_map(
    size: Tuple[int, int],
    keypoints: np.ndarray,
    kernel_size: int = 30,
    sigma: float = 8
) -> np.ndarray:
    """ function to generate density map from keypoints

    Args:
        size (Tuple[int, int]): size of the density map
        keypoints (np.ndarray): keypoints
        kernel_size (int, optional): size of the kernel. Defaults to 30.
        sigma (float, optional): sigma value. Defaults to 8.

    Returns:
        np.ndarray: density map
    """
    w, h = size
    keypoints = keypoints.astype(np.int32)
    density_map = np.zeros((h, w), dtype=np.float32)

    for keypoint in keypoints:
        keypoint = np.clip(keypoint, a_min=1, a_max=[w, h])
        x1, y1 = np.clip(np.array(keypoint - kernel_size // 2), a_min=1, a_max=[w, h])
        x2, y2 = np.clip(np.array(keypoint + kernel_size // 2), a_min=1, a_max=[w, h])
        gmap = gauss2D((y2 - y1 + 1, x2 - x1 + 1), sigma)
        density_map[y1 - 1:y2, x1 - 1:x2] = density_map[y1 - 1:y2, x1 - 1:x2] + gmap

    return density_map


def count_from_density_map(density_map: np.ndarray) -> int:
    """ counts number of keypoints from density map

    Args:
        density_map (np.ndarray): _description_

    Returns:
        int: number of keypoints
    """
    return round(np.sum(np.clip(density_map, a_min=0, a_max=None)))


def random_crop(
    img: Image.Image,
    density_map: np.ndarray,
    input_h: int = 512,
    input_w: int = 512
) -> Tuple[Image.Image, np.ndarray]:
    """ function for random crop image and density map"""
    img_w, img_h = img.size

    padded_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
    padded_density_map = np.zeros((input_h, input_w), dtype=np.float32)

    crop_min_x = np.random.randint(0, max(1, img_w - input_w))
    crop_min_y = np.random.randint(0, max(1, img_h - input_h))
    crop_max_x = min(crop_min_x + input_w, img_w)
    crop_max_y = min(crop_min_y + input_h, img_h)

    crop_w, crop_h = min(input_w, img_w), min(input_h, img_h)
    padded_img[:crop_h, :crop_w, :] = np.array(img)[crop_min_y:crop_max_y, crop_min_x:crop_max_x, :]
    padded_density_map[:crop_h, :crop_w] = density_map[crop_min_y:crop_max_y, crop_min_x:crop_max_x]

    return Image.fromarray(padded_img), padded_density_map


def resize_img(img: Image.Image, min_size: int = 512, max_size: int = 1536) -> Tuple[Image.Image, float]:
    """ function to resize image

    Args:
        img (Image.Image): image
        min_size (int, optional): minimum image size. Defaults to 512.
        max_size (int, optional): maximum image size. Defaults to 1536.

    Returns:
        Tuple[Image.Image, float]: resized image and scale
    """
    resize_ratio = 1.0
    if img.size[0] > max_size or img.size[1] > max_size:
        resize_ratio = max_size / img.size[0] if img.size[0] > max_size else max_size / img.size[1]
        resized_w = int(np.ceil(img.size[0] * resize_ratio))
        resized_h = int(np.ceil(img.size[1] * resize_ratio))
        img = img.resize((resized_w, resized_h))

    if img.size[0] < min_size or img.size[1] < min_size:
        resize_ratio = min_size / img.size[0] if img.size[0] < min_size else min_size / img.size[1]
        resized_w = int(np.ceil(img.size[0] * resize_ratio))
        resized_h = int(np.ceil(img.size[1] * resize_ratio))
        img = img.resize((resized_w, resized_h))

    return img, resize_ratio


def preprocess_transform(
    img: Image.Image,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> Tensor:
    """ function handles preprocessing of image by applying transforms

    Args:
        img (Image.Image): image
        mean (List[float], optional): mean for normalizing img. Defaults to [0.485, 0.456, 0.406].
        std (List[float], optional): standard deviation for normalizing img. Defaults to [0.229, 0.224, 0.225].

    Returns:
        Tensor: transformed image
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])(img)


def seed_worker(worker_id: int) -> None:
    """ function to seed random number generator for each worker"""
    worker_seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
