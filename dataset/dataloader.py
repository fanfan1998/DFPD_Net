from dataset.base_dataset import DeepfakeAbstractBaseDataset
from dataset.dfpd_dataset import DFPDDataset
import torch
import yaml

def prepare_training_data(config):
    # Only use the blending dataset class in training
    train_set = DFPDDataset(
        config=config,
        mode='train',
        # ae = True
    )
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            shuffle=True,
            num_workers=int(config['workers']),
            collate_fn=train_set.collate_fn,
        )
    return train_data_loader

def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = [test_name]  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test',
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=True
            )
        return test_data_loader
    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
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
