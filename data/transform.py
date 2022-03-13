from torchvision import transforms

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.GaussianBlur(3),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.3),
    transforms.Normalize((0.5,), (0.5,)),
])