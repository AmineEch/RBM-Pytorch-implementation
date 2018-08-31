from torchvision import datasets, transforms
import torch

def load_mnistdataloader(dataset_dir):
    '''
    Mnist dataloader function
    :return: dataloader
    '''
    mnist = datasets.MNIST(dataset_dir, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    data_loader = torch.utils.data.DataLoader(mnist, batch_size=128, shuffle=True, num_workers=1)
    return data_loader

