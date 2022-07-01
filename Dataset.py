# local lib
from CONFIG import DATASET_CONFIG
# public lib
import torch
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader

def image_Process(DATASET_CONFIG):

    '''
    Get the train_loader,validation_loader
    :param DATASET_CONFIG:
    :return: train_loader,validation_loader,classes_cnt,test_dataset
    '''

    # get the param of dataset
    train_dir = DATASET_CONFIG['train_dir']
    train_split = DATASET_CONFIG['train_split']
    validation_split = DATASET_CONFIG['validation_split']
    test_split = DATASET_CONFIG['test_split']
    batch_size = DATASET_CONFIG['batch_size']
    data_loading_workers = DATASET_CONFIG['data_loading_workers']
    random_seed = DATASET_CONFIG['random_seed']
    image_size = DATASET_CONFIG['image_size']
    resNext_mean = DATASET_CONFIG['resNext_mean']
    resNext_std = DATASET_CONFIG['resNext_std']

    image_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(resNext_mean,resNext_std)
        ]
    )

    dataset = ImageFolder(root=train_dir,transform=image_transform)
    # Get class count
    classes_cnt = len(dataset.classes)
    # Num of pic in training set
    train_size = int(len(dataset) * train_split)
    # Num of pic in dev set
    validation_size = int(len(dataset) * validation_split)
    # Num of pic in test set
    test_size = len(dataset) - train_size - validation_size

    # Split the DATASET
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size],
                                                                   generator=torch.Generator().manual_seed(random_seed))

    # Load training set
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=data_loading_workers, pin_memory=True
    )

    # Load dev set
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size,
        num_workers=data_loading_workers, pin_memory=True
    )


    return train_loader,validation_loader,classes_cnt,test_dataset


if __name__ == '__main__':
    pre_train_loader = None
    pre_dev_loader = None

    for i in range(300):
        train_loader, validation_loader, class_cnt, test_dataset = image_Process(DATASET_CONFIG)
        pre_dev_loader = validation_loader
        pre_train_loader = train_loader

    # print(f"Training set:{train_loader}")
    # print(f"Training set:{validation_loader}")