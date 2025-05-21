import os
import numpy as np
import cv2
import yaml
import re
import torch
from torch import nn
from utils.train_utils import load_dfpd_net
from tqdm import tqdm
from dataset.dataloader import prepare_testing_data
from utils.grad_cam_utils import GradCAM
import csv

def get_last_conv_name(net):
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    print('layer_name', layer_name)
    return layer_name


def gen_cam(image, mask):
    mask = np.uint8(255 * mask)
    mask_mean = float(mask.mean())
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(image)
    return norm_image(cam), (heatmap * 255).astype(np.uint8), mask_mean


def norm_image(image):
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)



def compute_iou(cam, mask_i, threshold=0.5):
    # Resize cam to match mask size
    if cam.shape != mask_i.shape:
        cam = cv2.resize(cam, (mask_i.shape[1], mask_i.shape[0]))
    # Normalize cam to [0, 1]
    cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    # Binarize
    cam_bin = cam_norm > threshold
    mask_bin = mask_i > 0.5
    intersection = np.logical_and(cam_bin, mask_bin).sum()
    union = np.logical_or(cam_bin, mask_bin).sum()
    iou = intersection / (union + 1e-6)
    return float(iou)



def compute_dice(cam, mask_i, threshold=0.5):
    # Resize cam to match mask size
    if cam.shape != mask_i.shape:
        cam = cv2.resize(cam, (mask_i.shape[1], mask_i.shape[0]))
    # Normalize cam to [0, 1]
    cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    # Binarize
    cam_bin = cam_norm > threshold
    mask_bin = mask_i > 0.5
    intersection = np.logical_and(cam_bin, mask_bin).sum()
    dice = (2. * intersection) / (cam_bin.sum() + mask_bin.sum() + 1e-6)
    return float(dice)


def gen_gb(grad):
    grad = grad.data.cpu().numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def sanitize_filename(name):
    return re.sub(r'[^\w\-_.]', '_', name)


def save_image(image_dicts, dir_name, prefix, draw_dir, is_correct, verbose=False):
    flag_folder = "correct" if is_correct else "wrong"
    save_root = os.path.join(draw_dir, sanitize_filename(dir_name), flag_folder)
    os.makedirs(save_root, exist_ok=True)
    for key in sorted(image_dicts.keys()):
        img = image_dicts[key]
        filename = f"{sanitize_filename(prefix)}_{key}.png"
        save_path = os.path.join(save_root, filename)
        cv2.imwrite(save_path, img)
        if verbose:
            print(f"[INFO] Saved: {save_path}")


def main():
    with open('visualize_config/cam_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['device'])

    print('load model from %s', config['model_name'])
    model_path = config['model_name']
    model = torch.load(model_path, map_location='cpu')
    dfpd_net = load_dfpd_net(model_name='resnet50', pretrain=True)
    dfpd_net.load_state_dict(model)
    dfpd_net = dfpd_net.to(device)
    dfpd_net.eval()

    if config['layer_name'] is None:
        layer_name = get_last_conv_name(dfpd_net)
    else:
        layer_name = config['layer_name']

    draw_dir = os.path.join(config['file_name'], "gradcam", layer_name)
    os.makedirs(draw_dir, exist_ok=True)
    print('draw dir: %s' % draw_dir)

    grad_cam = GradCAM(dfpd_net, layer_name, pred_key=config['pred_key'])
    max_total_images = config.get('max_total_images', float('inf'))
    total_image_count = 0
    draw_loader = prepare_testing_data(config)
    keys = draw_loader.keys()

    records = []

    for key in keys:
        total_images = sum(len(batch['image']) for batch in draw_loader[key])
        pbar = tqdm(total=min(total_images, max_total_images), desc=f"Processing {key}")
        for data_dict in draw_loader[key]:
            if total_image_count >= max_total_images:
                break
            inputs, labels, img_paths, masks = data_dict['image'], data_dict['label'], data_dict['image_path'], data_dict['mask']

            inputs = inputs.to(device)
            labels = labels.numpy()
            batch_size = inputs.shape[0]


            for i in range(batch_size):
                if total_image_count >= max_total_images:
                    break

                input_i = inputs[i].unsqueeze(0).requires_grad_(True)
                label_i = labels[i]
                img_path_i = img_paths[i]
                mask_i = masks[i].numpy()

                img_np = inputs[i].detach().cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * 0.5 + 0.5) * 255
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) / 255.0

                image_dict = {'original': np.uint8(img_bgr * 255), 'mask': np.uint8(255 * mask_i)}

                cam, pred_i = grad_cam(input_i, None, tuple(config['resize_size']))
                image_dict['cam'], _, mask_mean = gen_cam(img_bgr, cam)
                grad_cam.remove_handlers()

                save_image(
                    image_dicts=image_dict,
                    dir_name=img_path_i.split('/')[-5],
                    prefix=img_path_i.split('/')[-2] + '-' + os.path.splitext(os.path.basename(img_path_i))[0],
                    draw_dir=draw_dir,
                    is_correct=(label_i == pred_i),
                    verbose=False
                )

                records.append({
                    'method': img_path_i.split('/')[-5],
                    'image_path': img_path_i.split('/')[-2] + '-' + os.path.splitext(os.path.basename(img_path_i))[0],
                    'correct': int(label_i == pred_i),
                    'mask_mean': mask_mean,
                    'iou_score': compute_iou(cam, mask_i),
                    'dice_score': compute_dice(cam, mask_i)
                })

                total_image_count += 1
                pbar.update(1)
        pbar.close()

    csv_path = os.path.join(draw_dir, "gradcam_summary.csv")
    fieldnames = ['method', 'image_path', 'correct', 'mask_mean', 'iou_score', 'dice_score']
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"[INFO] Grad-CAM summary CSV saved to: {csv_path}")


if __name__ == '__main__':
    main()
