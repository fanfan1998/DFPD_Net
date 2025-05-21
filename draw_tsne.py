import os
import yaml
import numpy as np
from random import shuffle
from matplotlib import pyplot as plt
from openTSNE import TSNE
import torch
import torch.nn as nn
from utils.train_utils import load_dfpd_net
from dataset.dataloader import prepare_testing_data
from tqdm import tqdm

def tsne_vis(features, labels, draw_dir, config):
    embedding_path = os.path.join(draw_dir, '{}_embedding.npy'.format(config['feature_name']))
    img_path = os.path.join(draw_dir,  '{}_tsne.png'.format(config['feature_name']))
    print('tsne save path: %s' % img_path)
    
    if not os.path.exists(embedding_path):
        print(f">>> t-SNE fitting")
        tsne_model = TSNE(n_components=2, perplexity=config['perplexity'], random_state=1024, learning_rate=250)
        embeddings = tsne_model.fit(features)
        print(f"<<< fitting over")
        np.save(embedding_path, embeddings)        
    else:
        embeddings=np.load(embedding_path)

    index = [i for i in range(len(embeddings))]
    shuffle(index)
    embeddings = [embeddings[index[i]] for i in range(len(index))]
    labels = [labels[index[i]] for i in range(len(index))]
    embeddings = np.array(embeddings)
    print('embedding shape:', embeddings.shape)

    print(f">>> draw image")

    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]

    plt.figure(figsize=(12, 12))
    plt.rcParams['figure.dpi'] = 1000
    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    label_dict = {0: 'Real', 1: 'F2F', 2: 'DF', 3: 'FS', 4: 'NT'}

    unique_labels = set(labels)
    existing_labels = unique_labels.intersection(label_dict.keys())

    for label in existing_labels:
        color = colors[label] if label < len(colors) else 'tab:gray'
        class_index = [j for j, v in enumerate(labels) if v == label]
        marker = '*' if label == 0 else 'o'
        plt.scatter(vis_x[class_index], vis_y[class_index], c=color, s=10, alpha=0.9, label=label_dict[label],
                    marker=marker)

    if config['title'] is not None:
        plt.title(config['title'])
    plt.xticks([])
    plt.yticks([])
    plt.legend(fontsize=15)
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    print(f"<<<save image")

def extract_feature(model, draw_loader, device):
    pool = nn.AdaptiveAvgPool2d(1)
    model.eval()

    features_list = []
    labels_list = []

    with torch.no_grad():
        for data_dict in tqdm(draw_loader, desc="Extracting Features"):
            input_img, label = data_dict['image'], data_dict['domain_label']
            input_img = input_img.to(device)
            label = label.to(device)
            output_dict = model(input_img)
            feature = pool(output_dict[config['feature_name']]) if output_dict[config['feature_name']].dim() == 4 else output_dict[config['feature_name']]
            feature = feature.view(feature.shape[0], -1)
            features = feature.cpu().numpy()
            labels = label.cpu().numpy()

            features_list.append(features)
            labels_list.append(labels)

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    selected_features = []
    selected_labels = []

    unique_labels = np.unique(labels)
    print(unique_labels)
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        np.random.shuffle(indices)
        selected_indices = indices[:4000] if label == 0 else indices[:1000]
        selected_features.append(features[selected_indices])
        selected_labels.append(labels[selected_indices])

    selected_features = np.concatenate(selected_features, axis=0)
    selected_labels = np.concatenate(selected_labels, axis=0)

    return selected_features, selected_labels

if __name__ == '__main__':
    with open('visualize_config/tsne_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device(config['device'])

    print('load model from %s', config['model_name'])
    model_path = config['model_name']
    model = torch.load(model_path, map_location='cpu')
    dfpt_net = load_dfpd_net(model_name='resnet50', pretrain=False)
    dfpt_net.load_state_dict(model)
    dfpt_net = dfpt_net.to(device)

    draw_dir = os.path.join(config['file_name'], "tsne")
    os.makedirs(draw_dir, exist_ok=True)
    feature_path = os.path.join(draw_dir,  '{}_features.npy'.format(config['feature_name']))
    label_path = os.path.join(draw_dir,  '{}_labels.npy'.format(config['feature_name']))
    print('draw dir: %s' % draw_dir)

    draw_loader = prepare_testing_data(config)
    keys = draw_loader.keys()

    for key in keys:
        if not os.path.exists(feature_path):
            features, gt_labels = extract_feature(dfpt_net, draw_loader[key], device)
            np.save(feature_path, features)
            np.save(label_path, gt_labels)
        else:
            features = np.load(feature_path)
            gt_labels = np.load(label_path)

        print('labels:', gt_labels.shape, 'features:', features.shape)
        tsne_vis(features, gt_labels, draw_dir, config)
