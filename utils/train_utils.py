import logging
import numpy as np
import torch
from dfpd_net.core import DFPD_Net
from dfpd_net.resnet_backbone import resnet50
from utils.metric import get_test_metrics
import random
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU

    torch.backends.cudnn.benchmark = False      # 不自动寻找最优算法（以确保一致性）
    torch.backends.cudnn.deterministic = True

    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ Seed set to {seed} for full reproducibility.")

def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)

def load_dfpd_net(model_name, pretrain=True):
    print('==> Building model..')
    if model_name == 'resnet50':
        net = resnet50(pretrained=pretrain)
        net = DFPD_Net(net, 512)

    return net

def train_single_step(
    net, image, label, domain_label, mask,
    D1, D2, awl, pred, c_fea, f_fea,
    M_hat, Y_ce, Y_un, criterion, batch_size
):
    """
    单步训练逻辑封装函数。
    """
    output_dict = net(image)

    D1_out = D1(output_dict[c_fea])
    D2_out = D2(output_dict[f_fea][batch_size:])

    loss_cls = criterion["softmax"](output_dict[pred], label)
    loss_mask_val = criterion["mask"](output_dict[M_hat], mask)
    loss_pixel_val = criterion["pixel"](output_dict[Y_ce], output_dict[Y_un], mask)
    loss_content_adv1 = criterion["softmax"](D1_out, domain_label)
    loss_fake_adv2 = criterion["softmax"](D2_out, (domain_label[batch_size:] - 1))

    loss_total = awl(loss_cls, loss_mask_val, loss_pixel_val, loss_content_adv1, loss_fake_adv2)

    return loss_total, loss_cls, loss_mask_val, loss_pixel_val, loss_content_adv1, loss_fake_adv2, output_dict

def eval_single_step(net, criterion, testloader, device):
    net.eval()
    test_loss = 0
    idx = 0

    with torch.no_grad():
        total_pred = []
        total_label = []
        for batch_idx, data_dict in enumerate(testloader):
            image, label = data_dict['image'], data_dict['label']
            idx = batch_idx
            image, label = image.to(device), label.to(device)
            output_dict = net(image)
            output = output_dict['pred1'] + output_dict['pred2'] + output_dict['pred3']

            loss = criterion(output, label)
            test_loss += loss.item()

            pred = torch.softmax(output, dim=1)[:, 1]
            total_pred.append(pred.detach().squeeze().cpu().numpy())
            total_label.append(label.detach().squeeze().cpu().numpy())

    test_loss = test_loss / (idx + 1)
    metric = get_test_metrics(total_pred, total_label)

    return metric, test_loss

def log_losses(tag, loss_dict, batch_idx):

    print(
        f"Step: {batch_idx} | Loss1_{tag}: {loss_dict[0] / (batch_idx + 1):.5f} | "
        f"Loss2_{tag}: {loss_dict[1] / (batch_idx + 1):.5f} | "
        f"Loss3_{tag}: {loss_dict[2] / (batch_idx + 1):.5f} | "
        f"Loss_{tag}: {loss_dict[3] / (batch_idx + 1):.5f}"
    )

def log_epoch_info(phase, info_list):
    for info in info_list:
        logging.info(f'{phase}: {info}')
