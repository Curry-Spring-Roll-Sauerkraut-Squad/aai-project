import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

data_dir = '../../Dataset/Edited_Enhanced/'


simple_transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = ImageFolder(root=data_dir, transform=simple_transform)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

mean = torch.zeros(3)
std = torch.zeros(3)
num_samples = 0

for inputs, _ in data_loader:
    mean += torch.mean(inputs, dim=(0, 2, 3))
    std += torch.std(inputs, dim=(0, 2, 3))
    num_samples += 1

mean /= num_samples
std /= num_samples

print(f'Computed mean: {mean}')
print(f'Computed std: {std}')
