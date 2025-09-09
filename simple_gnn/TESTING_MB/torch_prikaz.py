import torch
# model = torch.load("../Dataset/criticality_dataset_nc1.pt", weights_only=False)
model = torch.load("./test_dataset.pth", weights_only=False)
for line in model:
        print(line)
        print()