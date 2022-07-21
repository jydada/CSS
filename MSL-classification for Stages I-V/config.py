class DefaultConfigs(object):
    #1.string parameters
    train_data = ".\\HE\\train\\"
    test_data = ".\\1M07_OR\\"
    val_data = ".\\HE\\val\\"
    model_name = "se_resnet50_HE"
    weights = ".\\checkpoints\\"
    best_models = weights + "best_model\\"
    submit = ".\\submit\\"
    logs = ".\\logs\\"
    gpus = "0"

    #2.numeric parameters
    epochs = 50
    batch_size = 3
    img_height = 1024
    img_weight = 1024
    crop_size = 512
    num_classes = 3
    seed = 888
    lr = 1e-5
    lr_decay = 1e-5
    weight_decay = 1e-5

config = DefaultConfigs()
