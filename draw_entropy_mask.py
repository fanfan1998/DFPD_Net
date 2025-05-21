import os
import numpy as np
import cv2
import yaml
import re
import torch
import torch.nn.functional as F
import csv
from tqdm import tqdm
from utils.train_utils import load_dfpd_net
from dataset.dataloader import prepare_testing_data


def sanitize_filename(name):
    return re.sub(r'[^\w\-_.]', '_', name)


def save_image(
    image_dicts,
    dir_name,
    prefix,
    draw_dir,
    is_correct=True,
    verbose=False
):

    flag_folder = "correct" if is_correct else "wrong"
    save_dir = os.path.join(draw_dir, sanitize_filename(dir_name), flag_folder)
    os.makedirs(save_dir, exist_ok=True)

    for key, img in image_dicts.items():
        filename = f"{sanitize_filename(prefix)}_{key}.png"
        image_path = os.path.join(save_dir, filename)

        # 兼容 float32 / float64 图像
        if img.dtype != 'uint8':
            img = (img * 255).astype('uint8') if img.max() <= 1 else img.astype('uint8')

        cv2.imwrite(image_path, img)
        if verbose:
            print(f"[Saved] {image_path}")


def visualize_entropy_E(net, inputs, image_dict, tags=['E1', 'E2', 'E3']):
    tag_pred_mean_dict = {}
    with torch.no_grad():
        output_dict = net(inputs)

        for tag in tags:
            if tag not in output_dict:
                continue

            suffix = tag[-1]  # 如 '1', '2', '3'
            pred_key = f'pred{suffix}'
            mhat_key = f'M_hat{suffix}'
            yce_key = f'Y_ce{suffix}'
            yun_key = f'Y_un{suffix}'

            if pred_key not in output_dict or mhat_key not in output_dict:
                continue

            # ---- 可视化熵图 E ----
            E = output_dict[tag]  # E1 / E2 / E3
            E_up = F.interpolate(E, size=inputs.shape[-2:], mode='bilinear', align_corners=False)
            E_np = E_up[0].squeeze().cpu().numpy()
            E_norm = (E_np - E_np.min()) / (E_np.max() - E_np.min() + 1e-6)
            E_map = np.uint8(E_norm * 255)
            E_map = np.ascontiguousarray(E_map, dtype=np.uint8)
            E_color = cv2.applyColorMap(E_map, cv2.COLORMAP_JET)
            image_dict[f'E{suffix}'] = E_color

            # ---- 可视化伪造概率图 M_hat ----
            M_hat = output_dict[mhat_key]
            M_up = F.interpolate(M_hat, size=inputs.shape[-2:], mode='bilinear', align_corners=False)
            M_np = M_up[0].squeeze().cpu().numpy()
            M_norm = (M_np - M_np.min()) / (M_np.max() - M_np.min() + 1e-6)
            M_map = np.uint8(M_norm * 255)
            M_map = np.ascontiguousarray(M_map, dtype=np.uint8)
            M_color = cv2.applyColorMap(M_map, cv2.COLORMAP_JET)
            image_dict[f'M_hat{suffix}'] = M_color

            # ---- 可视化 Y_ce ----
            if yce_key in output_dict:
                Y_ce = output_dict[yce_key]  # [1, C, H, W]
                Y_ce_up = F.interpolate(Y_ce, size=inputs.shape[-2:], mode='bilinear', align_corners=False)
                Y_ce_map = Y_ce_up[0].abs().sum(dim=0).cpu().numpy()  # [H, W]
                Y_ce_norm = (Y_ce_map - Y_ce_map.min()) / (Y_ce_map.max() - Y_ce_map.min() + 1e-6)
                Y_ce_vis = cv2.applyColorMap(np.uint8(Y_ce_norm * 255), cv2.COLORMAP_JET)
                image_dict[f'Y_ce{suffix}'] = Y_ce_vis

            # ---- 可视化 Y_un ----
            if yun_key in output_dict:
                Y_un = output_dict[yun_key]  # [1, C, H, W]
                Y_un_up = F.interpolate(Y_un, size=inputs.shape[-2:], mode='bilinear', align_corners=False)
                Y_un_map = Y_un_up[0].abs().sum(dim=0).cpu().numpy()  # [H, W]
                Y_un_norm = (Y_un_map - Y_un_map.min()) / (Y_un_map.max() - Y_un_map.min() + 1e-6)
                Y_un_vis = cv2.applyColorMap(np.uint8(Y_un_norm * 255), cv2.COLORMAP_JET)
                image_dict[f'Y_un{suffix}'] = Y_un_vis

            # ---- 分类预测值 ----
            pred_tensor = output_dict[pred_key]
            pred = pred_tensor.argmax(dim=1).item()
            entropy_mean = float(E_map.mean())
            tag_pred_mean_dict[tag] = {'pred': pred, 'mean': entropy_mean}

    return tag_pred_mean_dict



def main():
    with open('visualize_config/entropy_mask_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device(config['device'])

    print('load model from %s', config['model_name'])
    model_path = config['model_name']
    model = torch.load(model_path, map_location='cpu')
    dfpd_net = load_dfpd_net(model_name='resnet50', pretrain=True)
    dfpd_net.load_state_dict(model)
    dfpd_net = dfpd_net.to(device)
    dfpd_net.eval()

    draw_dir = os.path.join(config['file_name'], "entropymask", "multiE_tagwise")
    os.makedirs(draw_dir, exist_ok=True)
    print('draw dir: %s' % draw_dir)

    draw_loader = prepare_testing_data(config)
    entropy_tags = ['E1', 'E2', 'E3']
    max_total_images = config.get('max_total_images', float('inf'))
    total_image_count = 0
    entropy_records = []

    for key in draw_loader.keys():
        total_images = sum(len(batch['image']) for batch in draw_loader[key])
        pbar = tqdm(total=min(total_images, max_total_images), desc=f"Processing {key}")

        for batch_idx, data_dict in enumerate(draw_loader[key]):
            if total_image_count >= max_total_images:
                break
            inputs, labels, img_paths = data_dict['image'], data_dict['label'], data_dict['image_path']
            inputs = inputs.to(device)
            labels = labels.numpy()
            batch_size = inputs.shape[0]

            for i in range(batch_size):
                if total_image_count >= max_total_images:
                    break

                input_i = inputs[i].unsqueeze(0)
                label_i = labels[i]
                img_path_i = img_paths[i]

                img_np = inputs[i].detach().cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * 0.5 + 0.5) * 255
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                image_dict = {'image': img_bgr}
                tag_result_dict = visualize_entropy_E(dfpd_net, input_i, image_dict, tags=entropy_tags)

                for tag in tag_result_dict:
                    pred_i = tag_result_dict[tag]['pred']
                    entropy_mean = tag_result_dict[tag]['mean']
                    save_image(
                        image_dicts=image_dict,
                        dir_name=img_path_i.split('/')[-5],
                        prefix=img_path_i.split('/')[-2] + '-' + os.path.splitext(os.path.basename(img_path_i))[0],
                        draw_dir=draw_dir,
                        is_correct=(label_i == pred_i),
                        verbose=False
                    )
                    record = {
                        'method': img_path_i.split('/')[-5],
                        'image_path': img_path_i.split('/')[-2] + '-' + os.path.splitext(os.path.basename(img_path_i))[0],
                        'correct': int(label_i == pred_i),
                        f'entropy_{tag}_mean': entropy_mean
                    }
                    entropy_records.append(record)

                total_image_count += 1
                pbar.update(1)

        pbar.close()

    csv_path = os.path.join(draw_dir, "entropy_summary.csv")
    fieldnames = sorted(set().union(*(r.keys() for r in entropy_records)))
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in entropy_records:
            writer.writerow(row)
    print(f"[INFO] Entropy summary CSV saved to: {csv_path}")

if __name__ == '__main__':
    main()
