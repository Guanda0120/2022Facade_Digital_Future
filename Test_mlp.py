import torch
import os
from Dataset import image_Process
from CONFIG import DATASET_CONFIG,TRAIN_CONFIG

def model_Test(DATASET_CONFIG,TRAIN_CONFIG):

    '''
    return the test accuracy
    :param DATASET_CONFIG: config
    :return: none
    '''

    file_name = TRAIN_CONFIG['export_name']
    # check the right device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load temp model
    checkpoint = torch.load(os.path.join(r'.\out', file_name))
    # .pth file is a DICT
    # hub_repo is to get the original model dir
    hub_repo = checkpoint['hub_repo']
    # Get the model arch
    arch = checkpoint['arch']
    # Get classify num
    classes_cnt = checkpoint['classes_cnt']

    # Load model
    model = torch.hub.load(hub_repo, arch, pretrained=True)
    # Change the fc layer to num to classify
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features,model.fc.in_features),
        torch.nn.ReLU(),
        torch.nn.Linear(model.fc.in_features, model.fc.in_features),
        torch.nn.ReLU(),
        torch.nn.Linear(model.fc.in_features, classes_cnt),
    )

    # model.fc = torch.nn.Linear(model.fc.in_features, classes_cnt)
    # Pass the params
    model.load_state_dict(checkpoint['state_dict'])
    # Send to correct device
    model = model.to(device)

    # Load the test set
    _,_,_,test_dataset = image_Process(DATASET_CONFIG)

    # Correct num
    check = 0

    for image, target in test_dataset:
        image = image.to(device).unsqueeze(0)
        output = model(image)

        prediction = torch.argmax(output, dim=1).cpu().item()
        if prediction == target:
            check += 1

    print("{} %".format((check / len(test_dataset)) * 100))

if __name__ == '__main__':
    model_Test(DATASET_CONFIG,TRAIN_CONFIG)
