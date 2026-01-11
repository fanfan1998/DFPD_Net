from dataset.base_dataset import DeepfakeAbstractBaseDataset
from dataset.dfpd_dataset import DFPDDataset
import torch
import yaml

def prepare_training_data(config, rank=None, world_size=None):
    # Only use the blending dataset class in training
    train_set = DFPDDataset(
        config=config,
        mode='train',
        # ae = True
    )
    
    # 如果提供了 rank 和 world_size，使用分布式采样器
    if rank is not None and world_size is not None:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_set, num_replicas=world_size, rank=rank, shuffle=True
        )
        shuffle = False  # 使用 sampler 时不能 shuffle
        batch_size = config['train_batchSize'] // world_size  # 每个 GPU 的 batch size
    else:
        sampler = None
        shuffle = True
        batch_size = config['train_batchSize']
    
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=int(config['workers']),
            collate_fn=train_set.collate_fn,
        )
    return train_data_loader

def prepare_testing_data(config, rank=None, world_size=None):
    """
    准备测试数据加载器，支持分布式采样。
    
    Args:
        config: 配置字典
        rank: 当前进程的 rank（如果为 None，则使用单卡模式）
        world_size: 总 GPU 数量（如果为 None，则使用单卡模式）
    
    Returns:
        test_data_loaders: 测试数据加载器字典
    """
    def get_test_data_loader(config, test_name, rank=None, world_size=None):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = [test_name]  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test',
            )
        
        # 如果提供了 rank 和 world_size，使用分布式采样器
        if rank is not None and world_size is not None:
            sampler = torch.utils.data.distributed.DistributedSampler(
                test_set, num_replicas=world_size, rank=rank, shuffle=False
            )
            shuffle = False  # 使用 sampler 时不能 shuffle
        else:
            sampler = None
            shuffle = False
        
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=shuffle,
                sampler=sampler,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False  # 测试建议设为 False，保证覆盖所有样本
            )
        return test_data_loader
    
    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name, rank, world_size)
    return test_data_loaders

if __name__ == "__main__":
    with open('./dataset_config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_data_loader = prepare_training_data(config)
    test_data_loaders = prepare_testing_data(config)
    for batch_idx, data_dict in enumerate(train_data_loader):
        image, domain_label, label, mask = \
            data_dict['image'], data_dict['domain_label'], data_dict['label'], data_dict['mask']
        print(domain_label)

    # keys = test_data_loaders.keys()
    # for key in keys:
    #     for i, data_dict in enumerate(test_data_loaders[key]):
    #         data, label, mask, landmark = \
    #             data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
