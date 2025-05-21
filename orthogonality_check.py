import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from utils.train_utils import load_dfpd_net
from dataset.dataloader import prepare_testing_data
from tqdm import tqdm

def cosine_similarity(a, b):
    a_norm = F.normalize(a, dim=1)
    b_norm = F.normalize(b, dim=1)
    return (a_norm * b_norm).sum(dim=1).mean().item()

def pearson_corr(a, b):
    a_centered = a - a.mean(dim=1, keepdim=True)
    b_centered = b - b.mean(dim=1, keepdim=True)
    numerator = (a_centered * b_centered).sum(dim=1)
    denominator = torch.sqrt((a_centered ** 2).sum(dim=1)) * torch.sqrt((b_centered ** 2).sum(dim=1))
    return (numerator / (denominator + 1e-8)).mean().item()

def projection_energy_ratio(f_fea, c_fea, eps=1e-8):
    dot_prod = torch.sum(f_fea * c_fea, dim=1, keepdim=True)
    c_norm_sq = torch.sum(c_fea * c_fea, dim=1, keepdim=True).clamp(min=eps)
    f_proj = (dot_prod / c_norm_sq) * c_fea
    ratio = (f_proj.norm(p=2, dim=1) ** 2) / (f_fea.norm(p=2, dim=1) ** 2 + eps)
    return ratio.mean().item()

class MINE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(MINE, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))

def estimate_mutual_information(mine, x, y, optimizer, ma_et, ma_rate=0.1):
    y_shuffle = y[torch.randperm(y.size(0))]
    joint = mine(x, y)
    marginal = mine(x, y_shuffle)
    loss = - (joint.mean() - (torch.exp(marginal).mean().detach() * ma_et).log())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.exp(marginal).mean().detach()
    mi_est = joint.mean() - torch.log(torch.exp(marginal).mean() + 1e-8)
    return mi_est.item(), ma_et

def main():
    with open('visualize_config/orthogonality_check_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device(config['device'])

    print("[INFO] Loading model...")
    model_path = config['model_name']
    model = torch.load(model_path, map_location='cpu')
    dfpd_net = load_dfpd_net(model_name='resnet50', pretrain=True)
    dfpd_net.load_state_dict(model)
    dfpd_net = dfpd_net.to(device)
    dfpd_net.eval()

    draw_loader = prepare_testing_data(config)
    max_total_images = config.get('max_total_images', float('inf'))
    total_image_count = 0

    cosine_results = [[] for _ in range(3)]
    pearson_results = [[] for _ in range(3)]
    energy_ratios = [[] for _ in range(3)]
    mine_forgery = [[] for _ in range(3)]
    mine_content = [[] for _ in range(3)]

    cosine_results_dis = [[] for _ in range(3)]
    pearson_results_dis = [[] for _ in range(3)]
    energy_ratios_dis = [[] for _ in range(3)]
    mine_forgery_dis = [[] for _ in range(3)]
    mine_content_dis = [[] for _ in range(3)]

    print(f"[INFO] Max images to evaluate: {max_total_images}")
    for key in draw_loader:
        total_images = sum(len(batch['image']) for batch in draw_loader[key])
        pbar = tqdm(total=min(total_images, max_total_images), desc=f"Evaluating {key}")

        for batch in draw_loader[key]:
            if total_image_count >= max_total_images:
                break

            inputs = batch['image'].to(device)
            batch_size = inputs.shape[0]

            with torch.no_grad():
                output_dict = dfpd_net(inputs)

                for i in range(batch_size):
                    if total_image_count >= max_total_images:
                        break

                    for idx in range(3):
                        content = output_dict[f'c_fea{idx+1}'][i].unsqueeze(0)
                        forgery = output_dict[f'f_fea{idx+1}'][i].unsqueeze(0)
                        cos = cosine_similarity(forgery, content)
                        pcc = pearson_corr(forgery, content)
                        energy = projection_energy_ratio(forgery, content)

                        cosine_results[idx].append(cos)
                        pearson_results[idx].append(pcc)
                        energy_ratios[idx].append(energy)

                        mine_forgery[idx].append(forgery.squeeze(0).cpu())
                        mine_content[idx].append(content.squeeze(0).cpu())

                        dis = output_dict[f'f_dis{idx+1}'][i].unsqueeze(0)
                        cosine_results_dis[idx].append(cosine_similarity(dis, content))
                        pearson_results_dis[idx].append(pearson_corr(dis, content))
                        energy_ratios_dis[idx].append(projection_energy_ratio(dis, content))
                        mine_forgery_dis[idx].append(dis.squeeze(0).cpu())
                        mine_content_dis[idx].append(content.squeeze(0).cpu())

                    total_image_count += 1
                    pbar.update(1)

        pbar.close()

    print("\n[Original Forgery Feature Orthogonality & Mutual Information Results] (lower is better for orthogonality)\n")

    mi_results = []

    for stage in range(3):
        # Mutual Information
        f = torch.stack(mine_forgery[stage]).to(device)
        c = torch.stack(mine_content[stage]).to(device)
        mine = MINE(input_dim=f.shape[1]).to(device)
        optimizer = torch.optim.Adam(mine.parameters(), lr=1e-4)
        ma_et = torch.tensor(1.0).to(device)
        for _ in range(5):
            mi, ma_et = estimate_mutual_information(mine, f, c, optimizer, ma_et)
        mi_results.append(mi)

    # 输出所有指标
    for i in range(3):
        avg_cos = sum(cosine_results[i]) / len(cosine_results[i])
        avg_pcc = sum(pearson_results[i]) / len(pearson_results[i])
        avg_energy = sum(energy_ratios[i]) / len(energy_ratios[i])
        print(f"Stage {i + 1}:")
        print(f"  Cosine Similarity        : {avg_cos:.4f}")
        print(f"  Pearson Correlation      : {avg_pcc:.4f}")
        print(f"  Projection Energy Ratio  : {avg_energy:.4f}")
        print(f"  Mutual Information (MINE): {mi_results[i]:.4f}")
        print("")

    print("\n[Disentangled Forgery Feature Orthogonality & Mutual Information Results]\n")

    mi_results_dis = []
    for stage in range(3):
        f = torch.stack(mine_forgery_dis[stage]).to(device)
        c = torch.stack(mine_content_dis[stage]).to(device)
        mine = MINE(input_dim=f.shape[1]).to(device)
        optimizer = torch.optim.Adam(mine.parameters(), lr=1e-4)
        ma_et = torch.tensor(1.0).to(device)
        for _ in range(5):
            mi, ma_et = estimate_mutual_information(mine, f, c, optimizer, ma_et)
        mi_results_dis.append(mi)

    for i in range(3):
        avg_cos = sum(cosine_results_dis[i]) / len(cosine_results_dis[i])
        avg_pcc = sum(pearson_results_dis[i]) / len(pearson_results_dis[i])
        avg_energy = sum(energy_ratios_dis[i]) / len(energy_ratios_dis[i])
        print(f"Stage {i + 1}:")
        print(f"  Cosine Similarity        : {avg_cos:.4f}")
        print(f"  Pearson Correlation      : {avg_pcc:.4f}")
        print(f"  Projection Energy Ratio  : {avg_energy:.4f}")
        print(f"  Mutual Information (MINE): {mi_results_dis[i]:.4f}")
        print("")

if __name__ == '__main__':
    main()