from torch import nn
import torch.optim as optim
from dataset.dataloader import prepare_training_data, prepare_testing_data
import warnings
from loss.awl_loss import AutomaticWeightedLoss
from loss.lfd_loss import mask_loss as mask_loss_fn, pixel_loss as pixel_loss_fn
warnings.filterwarnings("ignore")
from utils.train_utils import *
from dfpd_net.discriminator import Discriminator
import yaml

def train(config):
    # setup output
    set_seed(config['seed'])
    store_dir = config['store_dir']
    device = torch.device(config['device'])

    try:
        os.stat(store_dir)
    except:
        os.makedirs(store_dir)

    use_cuda = torch.cuda.is_available()

    logging.basicConfig(
        filename=os.path.join(store_dir, 'record.log'),
        filemode='a',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO,
        force=True)

    # Data
    print('==> Preparing data..')
    train_data_loader = prepare_training_data(config)
    eval_data_loaders = prepare_testing_data(config)
    keys = eval_data_loaders.keys()

    max_iteration = len(train_data_loader) * config['nEpochs']
    print(f"Max Iteration Num: {max_iteration}")

    # Model
    net = load_dfpd_net(model_name='resnet50', pretrain=True).to(device)
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

    layers = [module for name, module in net.named_children() if name != 'features']
    layers.extend([criterion["awl1"], criterion["awl2"], criterion["awl3"],
                   S1_D1, S2_D1, S3_D1, S1_D2, S2_D2, S3_D2])

    params_lr_005 = [param for layer in layers for param in layer.parameters()]
    params_lr_0005 = list(net.features.parameters())

    # 构造优化器
    optimizer = optim.SGD([
        {'params': params_lr_005, 'lr': 0.005},
        {'params': params_lr_0005, 'lr': 0.0005}
    ],
        momentum=0.9, weight_decay=5e-4)

    max_val_AUC = 0

    for epoch in range(config['start_epoch'], config['nEpochs']):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()

        # 0:loss, 1:top-1, 2:EER, 3:HTER
        train_loss = [0, 0, 0, 0]

        loss_cls = [0, 0, 0, 0]
        loss_mask = [0, 0, 0, 0]
        loss_pixel = [0, 0, 0, 0]
        loss_content_adv1 = [0, 0, 0, 0]
        loss_fake_adv2 = [0, 0, 0, 0]

        # 0:correct, 1:total
        accuracy = [0, 0]

        for batch_idx, data_dict in enumerate(train_data_loader):

            image, domain_label, label, mask = data_dict['image'], data_dict['domain_label'], data_dict['label'], data_dict['mask']

            if mask is None:
                mask = label.view(-1, 1, 1, 1).expand(-1, 256, 256, 1).permute(0, 3, 1, 2).float()
            else:
                mask = mask.permute(0, 3, 1, 2).float()

            if image.shape[0] < config['train_batchSize'] * 2:
                continue
            if use_cuda:
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
                criterion, config['train_batchSize']
            )
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()

            # === Step2 ===
            loss2, loss2_cls, loss2_mask, loss2_pixel, loss2_content_adv1, loss2_fake_adv2, output_dict = train_single_step(
                net, image, label, domain_label, mask,
                S2_D1, S2_D2, criterion["awl2"],
                "pred2", "c_fea2", "f_fea2", "M_hat2", "Y_ce2", "Y_un2",
                criterion, config['train_batchSize']
            )
            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()

            # === Step3 ===
            loss3, loss3_cls, loss3_mask, loss3_pixel, loss3_content_adv1, loss3_fake_adv2, output_dict = train_single_step(
                net, image, label, domain_label, mask,
                S3_D1, S3_D2, criterion["awl3"],
                "pred3", "c_fea3", "f_fea3", "M_hat3", "Y_ce3", "Y_un3",
                criterion, config['train_batchSize']
            )
            optimizer.zero_grad()
            loss3.backward()
            optimizer.step()

            _, predicted = torch.max(output_dict['pred3'].data, 1)
            accuracy[1] += label.size(0)
            accuracy[0] += predicted.eq(label.data).cpu().sum()

            for i, val in enumerate([loss1, loss2, loss3]):
                train_loss[i] += val.item()
            train_loss[3] += sum(val.item() for val in [loss1, loss2, loss3])

            for i, val in enumerate([loss1_cls, loss2_cls, loss3_cls]):
                loss_cls[i] += val.item()
            loss_cls[3] += sum(val.item() for val in [loss1_cls, loss2_cls, loss3_cls])

            for i, val in enumerate([loss1_mask, loss2_mask, loss3_mask]):
                loss_mask[i] += val.item()
            loss_mask[3] += sum(val.item() for val in [loss1_mask, loss2_mask, loss3_mask])

            for i, val in enumerate([loss1_pixel, loss2_pixel, loss3_pixel]):
                loss_pixel[i] += val.item()
            loss_pixel[3] += sum(val.item() for val in [loss1_pixel, loss2_pixel, loss3_pixel])

            for i, val in enumerate([loss1_content_adv1, loss2_content_adv1, loss3_content_adv1]):
                loss_content_adv1[i] += val.item()
            loss_content_adv1[3] += sum(
                val.item() for val in [loss1_content_adv1, loss2_content_adv1, loss3_content_adv1])

            for i, val in enumerate([loss1_fake_adv2, loss2_fake_adv2, loss3_fake_adv2]):
                loss_fake_adv2[i] += val.item()
            loss_fake_adv2[3] += sum(val.item() for val in [loss1_fake_adv2, loss2_fake_adv2, loss3_fake_adv2])

            if batch_idx % 50 == 0:
                print(
                    f"Current Iteration: {(epoch - config['start_epoch']) * len(train_data_loader) + batch_idx + 1}/{max_iteration}")
                print(f"Current Epoch: {epoch + 1}/{config['nEpochs']}")

                print(
                    'Step: %d | Loss1: %.5f | Loss2: %.5f | Loss3: %.5f | Loss: %.5f | Acc: %.3f%% (%d/%d)' % (
                        batch_idx, train_loss[0] / (batch_idx + 1), train_loss[1] / (batch_idx + 1),
                        train_loss[2] / (batch_idx + 1), train_loss[3] / (batch_idx + 1),
                        100. * float(accuracy[0]) / accuracy[1], accuracy[0], accuracy[1]))

                log_losses("cls", loss_cls, batch_idx)
                log_losses("mask", loss_mask, batch_idx)
                log_losses("pixel", loss_pixel, batch_idx)
                log_losses("content_adv1", loss_content_adv1, batch_idx)
                log_losses("fake_adv2", loss_fake_adv2, batch_idx)

        # === Logging per-epoch information with timestamp ===
        log_epoch_info('Training', [
            'train dataset:%s' % str(config['train_dataset'])[1:-1],
            'epoch:%s' % (epoch + 1),
            'Domain_Accuracy:%.3f%%' % (100. * float(accuracy[0]) / accuracy[1]),
            'loss1:%.5f' % (train_loss[0] / (batch_idx + 1)),
            'loss2:%.5f' % (train_loss[1] / (batch_idx + 1)),
            'loss3:%.5f' % (train_loss[2] / (batch_idx + 1)),
            'train_loss:%.5f' % (train_loss[3] / (batch_idx + 1)),
        ])

        log_epoch_info('loss_cls', [
            'loss1_cls:%.5f' % (loss_cls[0] / (batch_idx + 1)),
            'loss2_cls:%.5f' % (loss_cls[1] / (batch_idx + 1)),
            'loss3_cls:%.5f' % (loss_cls[2] / (batch_idx + 1)),
            'loss_cls:%.5f' % (loss_cls[3] / (batch_idx + 1)),
        ])

        log_epoch_info('loss_mask', [
            'loss1_mask:%.5f' % (loss_mask[0] / (batch_idx + 1)),
            'loss2_mask:%.5f' % (loss_mask[1] / (batch_idx + 1)),
            'loss3_mask:%.5f' % (loss_mask[2] / (batch_idx + 1)),
            'loss_mask:%.5f' % (loss_mask[3] / (batch_idx + 1)),
        ])

        log_epoch_info('loss_pixel', [
            'loss1_pixel:%.5f' % (loss_pixel[0] / (batch_idx + 1)),
            'loss2_pixel:%.5f' % (loss_pixel[1] / (batch_idx + 1)),
            'loss3_pixel:%.5f' % (loss_pixel[2] / (batch_idx + 1)),
            'loss_pixel:%.5f' % (loss_pixel[3] / (batch_idx + 1)),
        ])

        log_epoch_info('loss_content_adv1', [
            'loss1_content_adv1:%.5f' % (loss_content_adv1[0] / (batch_idx + 1)),
            'loss2_content_adv1:%.5f' % (loss_content_adv1[1] / (batch_idx + 1)),
            'loss3_content_adv1:%.5f' % (loss_content_adv1[2] / (batch_idx + 1)),
            'loss_content_adv1:%.5f' % (loss_content_adv1[3] / (batch_idx + 1)),
        ])

        log_epoch_info('loss_fake_adv2', [
            'loss1_fake_adv2:%.5f' % (loss_fake_adv2[0] / (batch_idx + 1)),
            'loss2_fake_adv2:%.5f' % (loss_fake_adv2[1] / (batch_idx + 1)),
            'loss3_fake_adv2:%.5f' % (loss_fake_adv2[2] / (batch_idx + 1)),
            'loss_fake_adv2:%.5f' % (loss_fake_adv2[3] / (batch_idx + 1)),
        ])

        logging.info("")

        awl_logs = []
        for i, awl in enumerate([criterion["awl1"], criterion["awl2"], criterion["awl3"]], start=1):
            for param in awl.parameters():
                awl_logs.append(f'awl{i}_parameters: {param.cpu().detach().numpy()}')
        log_epoch_info('awl_parameters', awl_logs)

        for key in keys:
            eval_metric, eval_loss= eval_single_step(net, criterion["softmax"], eval_data_loaders[key], device)
            if eval_metric['auc'] >= max_val_AUC:
                max_val_AUC = eval_metric['auc']
                if config['save_model']:
                    torch.save(net.state_dict(), './' + store_dir + '/model.pth')

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

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train(config=config)