import torch
import os
from CONFIG import TRAIN_CONFIG

def export_Onnx(TRAIN_CONFIG):

    # Find file
    file_name = TRAIN_CONFIG['export_name']
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
        torch.nn.Linear(model.fc.in_features, model.fc.in_features),
        torch.nn.ReLU(),
        torch.nn.Linear(model.fc.in_features, model.fc.in_features),
        torch.nn.ReLU(),
        torch.nn.Linear(model.fc.in_features, classes_cnt),
    )
    # Pass the params
    model.load_state_dict(checkpoint['state_dict'])

    # switch to evaluation mode
    model.eval()
    # Let's create a dummy input tensor
    dummy_input = torch.randn(3, 224, 224, requires_grad=True).unsqueeze(0)

    # Export the model
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      os.path.join(r'.\out', file_name),  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],  # the model's input names
                      output_names=['modelOutput'],  # the model's output names
                      dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                    'modelOutput': {0: 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')


if __name__ == '__main__':
    export_Onnx(TRAIN_CONFIG)