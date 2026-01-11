import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import logging
import time
from datetime import datetime, timedelta
from torch import nn
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset.dataloader import prepare_training_data, prepare_testing_data
import warnings
from dfpd_net.dfpd_net import AutomaticWeightedLoss, mask_loss as mask_loss_fn, pixel_loss as pixel_loss_fn, Discriminator
warnings.filterwarnings("ignore")
from utils.train_utils import *
import yaml

def train(config, model_name='resnet50'):
    # torchrun 会自动设置这些环境变量
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # 初始化进程组
    # 设置 NCCL 超时时间（默认10分钟，这里设置为30分钟以应对可能的长时间评估）
    dist.init_process_group(
        backend='nccl',
        timeout=timedelta(minutes=30)
    )
    
    # 设置设备
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)
    
    # setup output
    set_seed(config['seed'])
    
    # 时间戳同步：只有 rank 0 生成，然后广播给其他进程
    if rank == 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        timestamp_tensor = torch.tensor([ord(c) for c in timestamp], dtype=torch.int, device=device)
    else:
        timestamp_tensor = torch.zeros(15, dtype=torch.int, device=device)  # YYYYMMDD_HHMMSS = 15 chars
    
    dist.broadcast(timestamp_tensor, src=0)
    timestamp = ''.join([chr(c) for c in timestamp_tensor.tolist()])
    
    # 设置输出目录：在 DFPD 后面添加时间戳
    base_dir = config['store_dir']
    if '/' in base_dir or '\\' in base_dir:
        # 分割路径，例如 'output/DFPD/FF++' -> ['output', 'DFPD', 'FF++']
        path_parts = base_dir.replace('\\', '/').split('/')
        # 找到 DFPD 的位置并添加时间戳
        if 'DFPD' in path_parts:
            dfpd_index = path_parts.index('DFPD')
            path_parts[dfpd_index] = 'DFPD_' + timestamp
            store_dir = '/'.join(path_parts)
        else:
            # 如果没有找到 DFPD，则在最后一个目录名后添加时间戳（保持原有逻辑）
            dir_path = os.path.dirname(base_dir)
            dir_name = os.path.basename(base_dir)
            store_dir = os.path.join(dir_path, dir_name + '_' + timestamp)
    else:
        store_dir = base_dir + '_' + timestamp
    
    # 计算每个 GPU 的实际 batch size
    local_batch_size = config['train_batchSize'] // world_size
    
    # 只在 rank 0 创建输出目录
    if rank == 0:
        os.makedirs(store_dir, exist_ok=True)
        print(f'Output directory: {store_dir}')
        print(f'DDP training with {world_size} GPUs')
    
    dist.barrier()
    
    use_cuda = torch.cuda.is_available()

    # 只在 rank 0 配置日志
    if rank == 0:
        logging.basicConfig(
            filename=os.path.join(store_dir, 'record.log'),
            filemode='a',
            format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
            level=logging.INFO,
            force=True)

    # Data
    if rank == 0:
        print('==> Preparing data..')
    train_data_loader = prepare_training_data(config, rank, world_size)
    eval_data_loaders = prepare_testing_data(config, rank, world_size)
    keys = eval_data_loaders.keys()

    max_iteration = len(train_data_loader) * config['nEpochs']
    if rank == 0:
        print(f"Max Iteration Num: {max_iteration}")

    # Model
    if rank == 0:
        print(f'==> Building model: {model_name}')
    net = load_dfpd_net(model_name=model_name, pretrain=True).to(device)
    S1_D1 = Discriminator(channels=1024, num_classes=5, alpha=10, max_iter=max_iteration).to(device)
    S2_D1 = Discriminator(channels=1024, num_classes=5, alpha=10, max_iter=max_iteration).to(device)
    S3_D1 = Discriminator(channels=1024, num_classes=5, alpha=10, max_iter=max_iteration).to(device)

    S1_D2 = Discriminator(channels=1024, num_classes=4, alpha=10, max_iter=max_iteration).to(device)
    S2_D2 = Discriminator(channels=1024, num_classes=4, alpha=10, max_iter=max_iteration).to(device)
    S3_D2 = Discriminator(channels=1024, num_classes=4, alpha=10, max_iter=max_iteration).to(device)

    criterion = {
        'awl1': AutomaticWeightedLoss(5).to(device),
        'awl2': AutomaticWeightedLoss(5).to(device),
        'awl3': AutomaticWeightedLoss(5).to(device),
        'softmax': nn.CrossEntropyLoss(),
        'mask': mask_loss_fn,
        'pixel': pixel_loss_fn,
    }

    # 使用 DDP 包装模型和判别器
    # 由于训练过程中每个步骤只使用部分输出参与损失计算（Step1用pred1相关，Step2用pred2相关，Step3用pred3相关），
    # 虽然前向传播计算了所有输出，但反向传播时只有部分参数参与梯度计算，因此需要设置 find_unused_parameters=True
    # 注意：这会产生警告，但这是必要的，因为每个步骤确实有未使用的参数
    net = DDP(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    S1_D1 = DDP(S1_D1, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    S2_D1 = DDP(S2_D1, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    S3_D1 = DDP(S3_D1, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    S1_D2 = DDP(S1_D2, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    S2_D2 = DDP(S2_D2, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    S3_D2 = DDP(S3_D2, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # 注意：DDP包装后需要使用module访问原始模型
    layers = [module for name, module in net.module.named_children() if name != 'features']
    layers.extend([criterion["awl1"], criterion["awl2"], criterion["awl3"],
                   S1_D1.module, S2_D1.module, S3_D1.module, S1_D2.module, S2_D2.module, S3_D2.module])

    params_lr_005 = [param for layer in layers for param in layer.parameters()]
    params_lr_0005 = list(net.module.features.parameters())

    # 构造优化器
    optimizer = optim.SGD([
        {'params': params_lr_005, 'lr': 0.005},
        {'params': params_lr_0005, 'lr': 0.0005}
    ],
        momentum=0.9, weight_decay=5e-4)

    max_val_AUC = 0
    # 记录训练开始时间
    train_start_time = time.time()

    for epoch in range(config['start_epoch'], config['nEpochs']):
        if hasattr(train_data_loader, 'sampler') and hasattr(train_data_loader.sampler, 'set_epoch'):
            train_data_loader.sampler.set_epoch(epoch)
        epoch_start_time = time.time()
        
        if rank == 0:
            print('\nEpoch: %d' % (epoch + 1))
        net.train()

        # 使用 GPU tensor 累积统计信息
        train_loss = [torch.tensor(0.0, device=device) for _ in range(4)]
        loss_cls = [torch.tensor(0.0, device=device) for _ in range(4)]
        loss_mask = [torch.tensor(0.0, device=device) for _ in range(4)]
        loss_pixel = [torch.tensor(0.0, device=device) for _ in range(4)]
        loss_content_adv1 = [torch.tensor(0.0, device=device) for _ in range(4)]
        loss_fake_adv2 = [torch.tensor(0.0, device=device) for _ in range(4)]

        # 0:correct, 1:total
        accuracy = [torch.tensor(0, device=device, dtype=torch.long), torch.tensor(0, device=device, dtype=torch.long)]

        for batch_idx, data_dict in enumerate(train_data_loader):

            image, domain_label, label, mask = data_dict['image'], data_dict['domain_label'], data_dict['label'], data_dict['mask']

            if mask is None:
                mask = label.view(-1, 1, 1, 1).expand(-1, 256, 256, 1).permute(0, 3, 1, 2).float()
            else:
                mask = mask.permute(0, 3, 1, 2).float()

            if image.shape[0] < local_batch_size * 2:
                continue
            
            # 移动到设备
            image = image.to(device)
            domain_label = domain_label.to(device)
            label = label.to(device)
            mask = mask.to(device)

            # 定义优化器的参数组数量
            num_param_groups = len(optimizer.param_groups)

            # 创建学习率列表lr，除了最后一个为0.0005，其余的都是0.005
            lr = [0.005] * (num_param_groups - 1) + [0.0005]

            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, config['nEpochs'], lr[nlr])

            # === Step1 ===
            loss1, loss1_cls, loss1_mask, loss1_pixel, loss1_content_adv1, loss1_fake_adv2, output_dict = train_single_step(
                net, image, label, domain_label, mask,
                S1_D1, S1_D2, criterion["awl1"],
                "pred1", "c_fea1", "f_fea1", "M_hat1", "Y_ce1", "Y_un1",
                criterion, local_batch_size
            )
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()

            # === Step2 ===
            loss2, loss2_cls, loss2_mask, loss2_pixel, loss2_content_adv1, loss2_fake_adv2, output_dict = train_single_step(
                net, image, label, domain_label, mask,
                S2_D1, S2_D2, criterion["awl2"],
                "pred2", "c_fea2", "f_fea2", "M_hat2", "Y_ce2", "Y_un2",
                criterion, local_batch_size
            )
            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()

            # === Step3 ===
            loss3, loss3_cls, loss3_mask, loss3_pixel, loss3_content_adv1, loss3_fake_adv2, output_dict = train_single_step(
                net, image, label, domain_label, mask,
                S3_D1, S3_D2, criterion["awl3"],
                "pred3", "c_fea3", "f_fea3", "M_hat3", "Y_ce3", "Y_un3",
                criterion, local_batch_size
            )
            optimizer.zero_grad()
            loss3.backward()
            optimizer.step()

            # GPU 上累积统计
            with torch.no_grad():
                _, predicted = torch.max(output_dict['pred3'].data, 1)
                accuracy[1] += label.size(0)
                accuracy[0] += predicted.eq(label.data).sum()

                for i, val in enumerate([loss1, loss2, loss3]):
                    train_loss[i] += val.detach()
                train_loss[3] += (loss1.detach() + loss2.detach() + loss3.detach())

                for i, val in enumerate([loss1_cls, loss2_cls, loss3_cls]):
                    loss_cls[i] += val.detach()
                loss_cls[3] += (loss1_cls.detach() + loss2_cls.detach() + loss3_cls.detach())

                for i, val in enumerate([loss1_mask, loss2_mask, loss3_mask]):
                    loss_mask[i] += val.detach()
                loss_mask[3] += (loss1_mask.detach() + loss2_mask.detach() + loss3_mask.detach())

                for i, val in enumerate([loss1_pixel, loss2_pixel, loss3_pixel]):
                    loss_pixel[i] += val.detach()
                loss_pixel[3] += (loss1_pixel.detach() + loss2_pixel.detach() + loss3_pixel.detach())

                for i, val in enumerate([loss1_content_adv1, loss2_content_adv1, loss3_content_adv1]):
                    loss_content_adv1[i] += val.detach()
                loss_content_adv1[3] += (loss1_content_adv1.detach() + loss2_content_adv1.detach() + loss3_content_adv1.detach())

                for i, val in enumerate([loss1_fake_adv2, loss2_fake_adv2, loss3_fake_adv2]):
                    loss_fake_adv2[i] += val.detach()
                loss_fake_adv2[3] += (loss1_fake_adv2.detach() + loss2_fake_adv2.detach() + loss3_fake_adv2.detach())

            # 打印进度
            if batch_idx % 50 == 0 and rank == 0:
                # 计算全局迭代数：每个epoch的总batch数 * epoch数 + 当前batch索引 + 1
                current_iteration = (epoch - config['start_epoch']) * len(train_data_loader) + batch_idx + 1
                print(
                    f"Current Iteration: {current_iteration}/{max_iteration}")
                print(f"Current Epoch: {epoch + 1}/{config['nEpochs']}")

                acc_percent = 100. * accuracy[0].item() / accuracy[1].item() if accuracy[1].item() > 0 else 0.0
                print(
                    'Step: %d | Loss1: %.5f | Loss2: %.5f | Loss3: %.5f | Loss: %.5f | Acc: %.3f%% (%d/%d)' % (
                        batch_idx, train_loss[0].item() / (batch_idx + 1), train_loss[1].item() / (batch_idx + 1),
                        train_loss[2].item() / (batch_idx + 1), train_loss[3].item() / (batch_idx + 1),
                        acc_percent, accuracy[0].item(), accuracy[1].item()), flush=True)

                # 转换 loss 列表为普通列表以便 log_losses 使用
                loss_cls_list = [l.item() / (batch_idx + 1) for l in loss_cls]
                loss_mask_list = [l.item() / (batch_idx + 1) for l in loss_mask]
                loss_pixel_list = [l.item() / (batch_idx + 1) for l in loss_pixel]
                loss_content_adv1_list = [l.item() / (batch_idx + 1) for l in loss_content_adv1]
                loss_fake_adv2_list = [l.item() / (batch_idx + 1) for l in loss_fake_adv2]
                
                log_losses("cls", loss_cls_list, batch_idx)
                log_losses("mask", loss_mask_list, batch_idx)
                log_losses("pixel", loss_pixel_list, batch_idx)
                log_losses("content_adv1", loss_content_adv1_list, batch_idx)
                log_losses("fake_adv2", loss_fake_adv2_list, batch_idx)

        # Epoch 结束统计
        epoch_time = time.time() - epoch_start_time
        
        # 统计每个 GPU 的实际 batch 数（考虑可能跳过的 batch）
        num_batches = torch.tensor(batch_idx + 1, device=device, dtype=torch.long)
        dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
        total_batches = num_batches.item()
        
        # 跨GPU聚合统计信息（所有GPU的loss和accuracy累加）
        for i in range(4):
            dist.all_reduce(train_loss[i], op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_cls[i], op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_mask[i], op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_pixel[i], op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_content_adv1[i], op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_fake_adv2[i], op=dist.ReduceOp.SUM)
        dist.all_reduce(accuracy[0], op=dist.ReduceOp.SUM)
        dist.all_reduce(accuracy[1], op=dist.ReduceOp.SUM)
        
        # 计算全局平均值（除以总batch数，而不是GPU数量）
        # 注意：train_loss 已经是所有 GPU 的累加值，需要除以总 batch 数
        if total_batches > 0:
            for i in range(4):
                train_loss[i] = train_loss[i] / total_batches
                loss_cls[i] = loss_cls[i] / total_batches
                loss_mask[i] = loss_mask[i] / total_batches
                loss_pixel[i] = loss_pixel[i] / total_batches
                loss_content_adv1[i] = loss_content_adv1[i] / total_batches
                loss_fake_adv2[i] = loss_fake_adv2[i] / total_batches
        
        # 只在 rank 0 进行日志记录和评估
        if rank == 0:
            # === Logging per-epoch information with timestamp ===
            acc_percent = 100. * accuracy[0].item() / accuracy[1].item() if accuracy[1].item() > 0 else 0.0
            log_epoch_info('Training', [
                'train dataset:%s' % str(config['train_dataset'])[1:-1],
                'epoch:%s' % (epoch + 1),
                'Bi_Accuracy:%.3f%%' % acc_percent,
                'loss1:%.5f' % train_loss[0].item(),
                'loss2:%.5f' % train_loss[1].item(),
                'loss3:%.5f' % train_loss[2].item(),
                'train_loss:%.5f' % train_loss[3].item(),
            ])

            log_epoch_info('loss_cls', [
                'loss1_cls:%.5f' % loss_cls[0].item(),
                'loss2_cls:%.5f' % loss_cls[1].item(),
                'loss3_cls:%.5f' % loss_cls[2].item(),
                'loss_cls:%.5f' % loss_cls[3].item(),
            ])

            log_epoch_info('loss_mask', [
                'loss1_mask:%.5f' % loss_mask[0].item(),
                'loss2_mask:%.5f' % loss_mask[1].item(),
                'loss3_mask:%.5f' % loss_mask[2].item(),
                'loss_mask:%.5f' % loss_mask[3].item(),
            ])

            log_epoch_info('loss_pixel', [
                'loss1_pixel:%.5f' % loss_pixel[0].item(),
                'loss2_pixel:%.5f' % loss_pixel[1].item(),
                'loss3_pixel:%.5f' % loss_pixel[2].item(),
                'loss_pixel:%.5f' % loss_pixel[3].item(),
            ])

            log_epoch_info('loss_content_adv1', [
                'loss1_content_adv1:%.5f' % loss_content_adv1[0].item(),
                'loss2_content_adv1:%.5f' % loss_content_adv1[1].item(),
                'loss3_content_adv1:%.5f' % loss_content_adv1[2].item(),
                'loss_content_adv1:%.5f' % loss_content_adv1[3].item(),
            ])

            log_epoch_info('loss_fake_adv2', [
                'loss1_fake_adv2:%.5f' % loss_fake_adv2[0].item(),
                'loss2_fake_adv2:%.5f' % loss_fake_adv2[1].item(),
                'loss3_fake_adv2:%.5f' % loss_fake_adv2[2].item(),
                'loss_fake_adv2:%.5f' % loss_fake_adv2[3].item(),
            ])

            logging.info("")

            awl_logs = []
            for i, awl in enumerate([criterion["awl1"], criterion["awl2"], criterion["awl3"]], start=1):
                for param in awl.parameters():
                    awl_logs.append(f'awl{i}_parameters: {param.cpu().detach().numpy()}')
            log_epoch_info('awl_parameters', awl_logs)

            if rank == 0:
                print('Epoch %d time: %.2f seconds' % (epoch + 1, epoch_time))

        # 评估阶段：所有 rank 都需要参与，因为 eval_single_step 内部有同步操作
        for key in keys:
            # 所有 rank 都参与评估计算（eval_single_step 内部有 dist.all_gather 等同步操作）
            eval_metric, eval_loss = eval_single_step(net, criterion["softmax"], eval_data_loaders[key], device, world_size)
            
            # 仅在 rank 0 打印和保存
            if rank == 0 and eval_metric is not None:
                if eval_metric['auc'] >= max_val_AUC:
                    max_val_AUC = eval_metric['auc']
                    if config['save_model']:
                        torch.save(net.module.state_dict(), './' + store_dir + '/model.pth')

                log_epoch_info('Evaling', [
                    'eval dataset:%s' % key,
                    'epoch:%s' % (epoch + 1),
                    'eval_loss:%.5f' % eval_loss,
                    'Acc:%.3f%%' % (eval_metric['acc'] * 100.0),
                    'AUC:%.5f' % eval_metric['auc'],
                    'EER:%.5f' % eval_metric['eer'],
                    'AP:%.5f' % eval_metric['ap'],
                ])

                logging.info("")
        
        dist.barrier()
    
    # 训练结束
    if rank == 0:
        total_time = time.time() - train_start_time
        print('Training completed! Total time: %.2f seconds' % total_time)
    
    dist.destroy_process_group()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, default=None)
    parser.add_argument('--test_dataset', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='resnet50', 
                        choices=['resnet50', 'xception'],
                        help='模型名称: resnet50 或 xception (默认: resnet50)')
    args = parser.parse_args()
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    if args.train_dataset:
        config['train_dataset'] = [ds.strip() for ds in args.train_dataset.split(',')]
    if args.test_dataset:
        config['test_dataset'] = [ds.strip() for ds in args.test_dataset.split(',')]
    
    train(config=config, model_name=args.model_name)