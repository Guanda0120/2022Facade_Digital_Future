# dataset config
DATASET_CONFIG = {
    # direct to image fold
    'train_dir':'./Metal',
    # portion of train,dev,test
    'train_split':0.8,
    'validation_split':0.1,
    'test_split':0.1,
    'batch_size':16,
    'data_loading_workers':2,
    'random_seed':42,
    'image_size':224,
    # pytorch resnext mean
    'resNext_mean':[0.485, 0.456, 0.406],
    # pytorch resnext std
    'resNext_std':[0.229, 0.224, 0.225]
}

# train config
TRAIN_CONFIG = {
    'hub_repo' : 'pytorch/vision:v0.10.0', # torch hub 储存库版本
    'model_arch' : 'resnext50_32x4d', # 模型架构
    'out_dir' : './out', # 存档储存位置
    'momentum' : 0.9, # 梯度下降动量
    'initial_learning_rate' : 5e-5, # 初始学习率
    'weight_decay' : 1e-3, # 梯度下降权值衰减
    'start_epoch' : 0, # 起始代数 方便做继续训练
    'epochs' : 10, # 终止代数
    'print_freq' : 100, # 每隔多少组数据输出一次日志
    'export_name' : 'Metal_10E.pth'
}

