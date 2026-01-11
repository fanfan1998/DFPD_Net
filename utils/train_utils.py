import logging
import numpy as np
import torch
import torch.distributed as dist
from dfpd_net.dfpd_net import DFPD_Net
from utils.metric import get_test_metrics
import random
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU

    torch.backends.cudnn.benchmark = True       # 启用自动寻找最优算法（提升性能）
    torch.backends.cudnn.deterministic = False  # 允许使用非确定性但更快的算法（提升性能）

    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ Seed set to {seed} (performance optimized mode).")

def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)

def load_dfpd_net(model_name='resnet50', pretrain=True):
    """
    加载 DFPD_Net 模型
    
    Args:
        model_name: 模型名称，默认'resnet50'
        pretrain: 是否使用预训练权重，默认True
    
    Returns:
        DFPD_Net 模型实例
    """
    print('==> Building model..')
    net = DFPD_Net(model=None, feature_size=512, model_name=model_name, pretrain=pretrain)
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

def eval_single_step(net, criterion, testloader, device, world_size=None):
    """
    分布式评估函数。
    每个 GPU 负责测试集的一部分，最后将所有 GPU 的预测结果和标签收集到一起计算最终指标。
    
    Args:
        net: 模型（DDP 包装后的模型，需要使用 net.module 访问原始模型）
        criterion: 损失函数
        testloader: 测试数据加载器
        device: 设备
        world_size: 总 GPU 数量（如果为 None，则使用单卡模式）
    
    Returns:
        metric: 评估指标字典（仅在 rank 0 返回，其他 rank 返回 None）
        test_loss: 平均测试损失（仅在 rank 0 返回，其他 rank 返回 None）
    """
    net.eval()
    test_loss = torch.tensor(0.0, device=device)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data_dict in testloader:
            image, label = data_dict['image'].to(device), data_dict['label'].to(device)
            
            # 如果 net 是 DDP 包装的，使用 net.module；否则直接使用 net
            if hasattr(net, 'module'):
                output_dict = net.module(image)
            else:
                output_dict = net(image)
            
            # 三个预测分支融合
            output = output_dict['pred1'] + output_dict['pred2'] + output_dict['pred3']
            
            loss = criterion(output, label)
            test_loss += loss.detach()

            # 提取概率，转为 float32 以便 gather
            pred = torch.softmax(output, dim=1)[:, 1]
            all_preds.append(pred)
            all_labels.append(label)

    # 如果 world_size 为 None 或小于等于 1，使用单卡模式
    if world_size is None or world_size <= 1:
        # 单卡模式：直接计算指标
        if len(all_preds) > 0:
            final_preds = torch.cat(all_preds).cpu().numpy()
            final_labels = torch.cat(all_labels).cpu().numpy()
            num_batches = len(testloader) if len(testloader) > 0 else 1
            avg_loss = test_loss.item() / num_batches
            metric = get_test_metrics([final_preds], [final_labels])
        else:
            # 如果没有数据，返回空指标
            avg_loss = 0.0
            metric = {'auc': 0.0, 'eer': 1.0, 'ap': 0.0, 'acc': 0.0}
        return metric, avg_loss
    
    # 分布式模式：收集所有 GPU 的结果
    # 拼接当前 GPU 上的所有 batch
    local_preds = torch.cat(all_preds) if len(all_preds) > 0 else torch.tensor([], device=device, dtype=torch.float32)
    local_labels = torch.cat(all_labels) if len(all_labels) > 0 else torch.tensor([], device=device, dtype=torch.long)

    # 统计每个 GPU 的实际 batch 数
    local_num_batches = torch.tensor(len(all_preds), device=device, dtype=torch.long)
    num_batches_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(num_batches_list, local_num_batches.unsqueeze(0))
    total_batches = sum([nb.item() for nb in num_batches_list])

    # 获取每个 GPU 的数据大小，以便处理不同大小的数据
    local_size = torch.tensor([local_preds.size(0)], device=device, dtype=torch.long)
    sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    sizes = [s.item() for s in sizes]
    max_size = max(sizes) if max(sizes) > 0 else 1

    # 如果当前 GPU 的数据量小于最大数据量，需要 padding（all_gather 要求所有 tensor 大小相同）
    if local_preds.size(0) < max_size:
        padding_size = max_size - local_preds.size(0)
        pred_padding = torch.zeros(padding_size, device=device, dtype=local_preds.dtype)
        label_padding = torch.zeros(padding_size, device=device, dtype=local_labels.dtype) - 1  # 使用 -1 作为无效标签
        local_preds = torch.cat([local_preds, pred_padding])
        local_labels = torch.cat([local_labels, label_padding])
    elif local_preds.size(0) == 0:
        # 如果当前 GPU 完全没有数据，创建填充 tensor
        pred_padding = torch.zeros(max_size, device=device, dtype=torch.float32)
        label_padding = torch.zeros(max_size, device=device, dtype=torch.long) - 1
        local_preds = pred_padding
        local_labels = label_padding

    # 准备全局容器（List of Tensors）
    gathered_preds = [torch.zeros_like(local_preds) for _ in range(world_size)]
    gathered_labels = [torch.zeros_like(local_labels) for _ in range(world_size)]

    # 全局同步：把所有 GPU 的结果拉到每个 GPU 上
    dist.all_gather(gathered_preds, local_preds)
    dist.all_gather(gathered_labels, local_labels)
    dist.all_reduce(test_loss, op=dist.ReduceOp.SUM)

    # 仅在主进程计算最终指标
    if dist.get_rank() == 0:
        # 移除 padding 并合并所有 GPU 的结果
        final_preds_list = []
        final_labels_list = []
        for i in range(world_size):
            if sizes[i] > 0:  # 只处理有数据的 GPU
                pred = gathered_preds[i][:sizes[i]].cpu().numpy()
                label = gathered_labels[i][:sizes[i]].cpu().numpy()
                # 过滤掉无效标签（padding 的 -1）
                valid_mask = label >= 0
                if valid_mask.sum() > 0:
                    final_preds_list.append(pred[valid_mask])
                    final_labels_list.append(label[valid_mask])
        
        # 计算全局平均 loss：test_loss 已经累加了所有 GPU 的 loss，需要除以总 batch 数
        total_avg_loss = test_loss.item() / total_batches if total_batches > 0 else 0.0
        
        # 调用指标计算工具
        if len(final_preds_list) > 0:
            metric = get_test_metrics(final_preds_list, final_labels_list)
        else:
            metric = {'auc': 0.0, 'eer': 1.0, 'ap': 0.0, 'acc': 0.0}
        return metric, total_avg_loss
    
    # 非主进程返回 None，防止重复打印
    return None, None

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
