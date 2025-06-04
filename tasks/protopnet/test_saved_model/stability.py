import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from tasks.protopnet.preprocessing.preprocess_protopnet import mean, std

def stability_score(model, dataloader, device='cuda', max_batches=50):
    model.eval()
    sims = []

    with torch.no_grad():
        for i, (imgs, _) in enumerate(tqdm(dataloader, desc="Evaluating stability")):
            if i >= max_batches:
                break

            imgs = imgs.to(device)

            perturbed_imgs = imgs + 0.01 * torch.randn_like(imgs)
            perturbed_imgs = torch.clamp(perturbed_imgs, 0, 1)

            _, dists_orig = model.push_forward(imgs)
            _, dists_perturbed = model.push_forward(perturbed_imgs)

            batch_size, num_prototypes, H, W = dists_orig.shape
            d1 = dists_orig.view(batch_size, num_prototypes, -1).mean(dim=2)
            d2 = dists_perturbed.view(batch_size, num_prototypes, -1).mean(dim=2)

            a1 = -d1
            a2 = -d2

            a1_norm = F.normalize(a1, dim=1)
            a2_norm = F.normalize(a2, dim=1)

            sim = torch.sum(a1_norm * a2_norm, dim=1)
            sims.extend(sim.cpu().numpy())

    sims = np.array(sims)
    return sims.mean(), sims.std()

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()

    model_path =  r"D:\Facultate\Auto\explainable_ai\saved_models\resnet34\experiment_run\7nopush0.7301.pth"
    model = torch.load(model_path)
    model = model.cuda()
    model.eval()

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_root = Path(r"D:\Facultate\Auto\explainable_ai\tasks\protopnet\datasets\cub200_cropped\test_cropped")
    val_ds = datasets.ImageFolder(val_root, transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

    is_wrapper = hasattr(model, 'core')
    core_model = model.core if is_wrapper else model

    mean_sim, std_sim = stability_score(core_model, val_loader, device="cuda", max_batches=50)
    print(f"\n=== Explanation Stability (cosine similarity): {mean_sim:.3f} ± {std_sim:.3f} ===")
