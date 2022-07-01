import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils
import torch


def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    n, c, w, h = tensor.shape

    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.ioff()


if __name__ == "__main__":
    layer = 1
    model = torch.load(r"D:\LiGD\NoahDF\DF_transferLearning\out\Metal_10E.pth")
    for key in model["state_dict"].keys():
        if 'conv' in key:
            filter = model["state_dict"][key].data.clone().cpu()
            visTensor(filter, ch=0, allkernels=False)
            plt.savefig(os.path.join('C:\\Users\\xkool1\\Desktop\\convfig',('.'.join([key,'png']))))



