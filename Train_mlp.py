from CONFIG import TRAIN_CONFIG,DATASET_CONFIG
from Dataset import image_Process
import os
import torch
from torch.utils.tensorboard import SummaryWriter

def training_Process(TRAIN_CONFIG,DATASET_CONFIG):
    '''
    a training function, save .pth model in out fold
    :param TRAIN_CONFIG: config
    :param DATASET_CONFIG: confg
    :return: a temp model
    '''
    # check could use cuda or not
    # make sure that install the cuda, cuDNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # init a tensorboard
    tensor_board = SummaryWriter()

    # Get hyperparam from config
    hub_repo = TRAIN_CONFIG['hub_repo']
    model_arch = TRAIN_CONFIG['model_arch']
    out_dir = TRAIN_CONFIG['out_dir']
    momentum = TRAIN_CONFIG['momentum']
    initial_learning_rate = TRAIN_CONFIG['initial_learning_rate']
    weight_decay = TRAIN_CONFIG['weight_decay']
    start_epoch = TRAIN_CONFIG['start_epoch']
    epochs = TRAIN_CONFIG['epochs']
    print_freq = TRAIN_CONFIG['print_freq']
    export_name = TRAIN_CONFIG['export_name']

    # load the class num set
    _,_,classes_cnt,_ = image_Process(DATASET_CONFIG)

    # change the architecture of fc
    # Loading the pre trained resNext
    model = torch.hub.load(hub_repo, model_arch, pretrained=True)
    # attribute model.fc.in_features means input how much nuera_node
    # change to dataset class
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features,model.fc.in_features),
        torch.nn.ReLU(True),
        torch.nn.Linear(model.fc.in_features, model.fc.in_features),
        torch.nn.ReLU(True),
        torch.nn.Linear(model.fc.in_features, classes_cnt)
    )

    #model.fc = torch.nn.Linear(model.fc.in_features, classes_cnt)
    # xavier init the weight of mlp
    for ly in model.fc:
        if isinstance(ly, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(ly.weight)

    # set to correct device
    model = model.to(device)

    # Set the optimizer
    # Set the params need to optimize
    params = []
    for name,para in model.named_parameters():
        if name not in ['fc.0.weight','fc.0.bias','fc.2.weight','fc.2.bias','fc.4.weight','fc.4.bias']:
            params.append(para)

    # The conv net should update para slowly, and new layer should fast converge
    optimizer = torch.optim.SGD([{'params': params,}, {'params': model.fc.parameters(), 'lr': initial_learning_rate * 10}],
                                lr=initial_learning_rate, momentum=momentum, weight_decay=weight_decay)
    # Set loss Func
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # Dynamic lr
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # ========================== Start Training ======================================
    for epoch in range(start_epoch, epochs):
        print("--------------------------TRAINING--------------------------------")
        # Switch to training mode
        model.train()
        train_data, dev_data, classes_cnt, _ = image_Process(DATASET_CONFIG)

        for i, (images, labels) in enumerate(train_data):
            # Send a tensor of image to CUDA，shape([batch_size, 3, image_size, image_size])
            images = images.to(device)
            # Send a tensor of label to CUDA，shape([batch_size])
            labels = labels.to(device)

            # Forward Propagate, and compute loss
            output = model(images)
            train_loss = criterion(output, labels)

            # Index out the predition label
            prediction = torch.argmax(output, dim=1)
            # Compute the correct prediction num
            train_correct = (prediction == labels).sum()
            # Compute Accurate ratio
            train_acc = (train_correct.float()) / images.shape[0]

            # Print acc
            if i % print_freq == 0:
                print("[Epoch {:02d}] ({:03d}/{:03d}) | Loss: {:.18f} | ACC: {:.2f} %".format(epoch, i + 1,
                                                                                              len(train_data),
                                                                                              train_loss.item(),
                                                                                              train_acc * 100))

            # Zero Grad
            optimizer.zero_grad()
            # Back Prop
            train_loss.backward()
            # Optimze
            optimizer.step()

        # Add to tensorboard
        tensor_board.add_scalar("Train Loss", train_loss, epoch)
        tensor_board.add_scalar("Train Correct", train_correct, epoch)
        tensor_board.add_scalar("Train Accuracy", train_acc, epoch)


        print("--------------------------DEV--------------------------------")
        with torch.no_grad():

            for j, (images,labels) in enumerate(dev_data):
                # Send a tensor of image to CUDA，shape([batch_size, 3, image_size, image_size])
                images = images.to(device)
                # Send a tensor of label to CUDA，shape([batch_size])
                labels = labels.to(device)

                # Forward Propagate, and compute loss
                output = model(images)
                dev_loss = criterion(output, labels)

                # Index out the predition label
                dev_prediction = torch.argmax(output, dim=1)
                # Compute the correct prediction num
                dev_correct = (dev_prediction == labels).sum()
                # Compute Accurate ratio
                dev_acc = (dev_correct.float()) / images.shape[0]

                # print result
                if j % print_freq == 0:
                    print("[Epoch {:02d}] ({:03d}/{:03d}) | Loss: {:.18f} | ACC: {:.2f} %".format(epoch, j + 1,
                                                                                                  len(dev_data),
                                                                                                  dev_loss.item(),
                                                                                                  dev_acc * 100))

        # TensorBoard
        tensor_board.add_scalar("Val Loss", dev_loss, epoch)
        tensor_board.add_scalar("Val Correct", dev_correct, epoch)
        tensor_board.add_scalar("Val Accuracy", dev_acc, epoch)

        # Refresh the lr
        scheduler.step()

        # Save model for this epoch
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, export_name)
        state = {
            'epoch': epoch + 1,
            'arch': model_arch,
            'hub_repo': hub_repo,
            'state_dict': model.state_dict(),
            'classes_cnt': classes_cnt
        }

        torch.save(state, filename)


if __name__ == '__main__':

    training_Process(TRAIN_CONFIG, DATASET_CONFIG)
