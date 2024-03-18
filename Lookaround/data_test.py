import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import defaultdict

def split_dataset_by_class(dataset, train_ratio=0.8):
    class_to_samples = defaultdict(list)
    for i in range(len(dataset)):
        img, label = dataset[i]
        class_to_samples[label].append(i)

    train_indices = []
    valid_indices = []
    for class_idx, samples in class_to_samples.items():
        num_samples = len(samples)
        num_train_samples = int(train_ratio * num_samples)
        train_indices.extend(samples[:num_train_samples])
        valid_indices.extend(samples[num_train_samples:])

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    valid_subset = torch.utils.data.Subset(dataset, valid_indices)

    return train_subset, valid_subset

# 数据集的存储路径
data_path = "./dataset/"

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 获取 CIFAR-100 数据集
full_dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)

# 划分训练集和验证集
train_dataset, valid_dataset = split_dataset_by_class(full_dataset, train_ratio=0.8)

# 获取 CIFAR-100 测试集
test_dataset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)

from collections import Counter

# 统计训练集中各类别的样本数量
train_class_counts = Counter([train_dataset[i][1] for i in range(len(train_dataset))])

# 统计验证集中各类别的样本数量
valid_class_counts = Counter([valid_dataset[i][1] for i in range(len(valid_dataset))])

# 打印训练集和验证集中各类别的样本数量
print("训练集中各类别的样本数量：")
for class_idx, count in train_class_counts.items():
    print(f"类别 {class_idx}: {count} 个样本")

print("\n验证集中各类别的样本数量：")
for class_idx, count in valid_class_counts.items():
    print(f"类别 {class_idx}: {count} 个样本")
