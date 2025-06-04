import torch
from PIL import Image
from torchvision import transforms
from tasks.protopnet.preprocessing.preprocess_protopnet import mean, std
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F

# model_path = r"D:\Facultate\Auto\explainable_ai\saved_models\resnet34\experiment_run\7nopush0.7301.pth"
# img = Image.open(r"D:\Facultate\Auto\explainable_ai\tasks\protopnet\datasets\cub200_cropped\train_cropped\002.Laysan_Albatross\Laysan_Albatross_0003_1033.jpg").convert("RGB")
# dataset_root = r"D:\Facultate\Auto\explainable_ai\tasks\protopnet\datasets\cub200_cropped\train_cropped"

model_path = r"D:\Facultate\Auto\explainable_ai\saved_models\protopnet_cars_cropped\resnet34\experiment_run\2nopush0.7164.pth"
img = Image.open(r"D:\Facultate\Auto\explainable_ai\tasks\protopnet\datasets\stanford_cars_cropped\train_cropped\Volvo XC90 SUV 2007\00213.jpg").convert("RGB")
dataset_root = r"D:\Facultate\Auto\explainable_ai\tasks\protopnet\datasets\stanford_cars_cropped\train_cropped"

model = torch.load(model_path)
model = model.cuda()
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

img_tensor = transform(img).unsqueeze(0).cuda()

with torch.no_grad():
    logits, _ = model(img_tensor)
    predicted_class = torch.argmax(logits, dim=1).item()
    print("Predicted class index:", predicted_class)

class_names = datasets.ImageFolder(dataset_root).classes

print("Predicted class name:", class_names[predicted_class])

_, distances = model.push_forward(img_tensor)
min_dists = torch.min(distances.view(model.num_prototypes, -1), dim=1)[0]
topk_proto_indices = torch.topk(-min_dists, k=5).indices.cpu().numpy()
print("Top activated prototypes:", topk_proto_indices)

conv_output, distances = model.push_forward(img_tensor)

most_activated_proto = topk_proto_indices[0]
activation_map = -distances[0, most_activated_proto]  # Shape: (H, W), negative distance = higher activation

activation_map = activation_map - activation_map.min()
activation_map = activation_map / activation_map.max()

activation_map = activation_map.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
activation_map = F.interpolate(activation_map, size=(224, 224), mode='bilinear', align_corners=False)
activation_map = activation_map.squeeze().cpu().detach().numpy()


img_np = transforms.ToTensor()(img.resize((224, 224))).permute(1, 2, 0).numpy()

plt.figure(figsize=(6, 6))
plt.imshow(img_np)
plt.imshow(activation_map, cmap='jet', alpha=0.5)  # Overlay heatmap
plt.title(f"Most activated prototype: {most_activated_proto}")
plt.axis('off')
plt.tight_layout()
plt.show()
