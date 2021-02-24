import torch
from torch.utils.data import Dataset, random_split, DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from argparse import Namespace



# custom Dataset
class DogData(Dataset) :
    def __init__(self, ds, transform = None) :
        self.ds = ds
        self.transform = transform
    
    def __len__(self) :
        return len(self.ds)
    
    def __getitem__(self, idx) :
        img, label = self.ds[idx]
        if self.transform :
            img = self.transform(img)
            return img, label


def dataloader(args) : 

    # another
    DATA_PATH = args.DATA_PATH
    dataset = ImageFolder(DATA_PATH) 

    test_pct = args.test_pct 
    test_size = int(len(dataset)*test_pct)
    dataset_size = len(dataset) - test_size

    val_pct = args.val_pct
    val_size = int(dataset_size*val_pct)
    train_size = dataset_size - val_size

    train, val, test = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # arg   
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225]) 
                                        ])

    val_transforms = transforms.Compose([
                                        transforms.Resize(255), 
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([
                                        transforms.Resize(255), 
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    train = DogData(train, train_transforms)
    val = DogData(val, val_transforms)
    test = DogData(test, test_transforms)

    num_workers = args.num_workers
    batch_size = args.batch_size

    trainLoader = torch.utils.data.DataLoader(train, batch_size=batch_size, 
                                                num_workers=num_workers, shuffle=True)
    valLoader = torch.utils.data.DataLoader(val, batch_size=batch_size, 
                                                num_workers=num_workers, shuffle=True)
    testLoader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                                num_workers=num_workers, shuffle=True)
    
    return trainLoader, valLoader, testLoader