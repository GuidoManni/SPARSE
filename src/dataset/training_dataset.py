import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


def get_loader(root_path, batch_size=32, transform=None, num_workers=4, shuffle=True):
    """
    Create a DataLoader for the dataset with the given structure.

    Args:
    - root_path (str): Path to the root directory of the dataset.
    - batch_size (int): Number of images to process in a batch.
    - transform (torchvision.transforms.Compose): Transformations to apply to the images.
    - num_workers (int): Number of subprocesses to use for data loading.
    - shuffle (bool): Whether to shuffle the dataset.

    Returns:
    - DataLoader: DataLoader object for the dataset.
    """
    if transform is None:
        # Default transformation: Resize images to 256x256 and convert them to tensor
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    # Load dataset using ImageFolder
    dataset = datasets.ImageFolder(root=root_path, transform=transform)

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)

    return loader





if __name__ == "__main__":
    root_dataset_folder = "/home/gvide/Dataset/UCG/Pneumonia/training"  # Replace with the path to your dataset
    batch_size = 32

    dataloader = get_loader(root_path=root_dataset_folder, batch_size=batch_size)

    # Example of iterating through the DataLoader
    for images, labels in dataloader:
        print(images.shape, labels)
        break  # Just show the first batch


