class DefaultConfigs(object):

    seed = 2022
    comment = 'FG 2021.'
    log_name = "Mar_0" # log's name 
    gpus = '2'
    model = 'xception'
    singleside = False
    rsc = False
    lq = False
    aug = True

    lr = 1e-4
    weight_decay = 1e-4
    beta1 = 0.9
    beta2 = 0.999
    num_classes = 2
    pretrained = True
    pretrained_weights = None
    
    batch_size = 16          # batch size: n * 2
    max_iter = 500000        # To be modified according to data size
    iter_per_epoch = 50
    iter_per_eval = 50
    save_freq = 10
    backbone_pretrain = '../pretrain/xception-c0a72b38.pth.tar'

    # paths information
    checkpoint_path = './checkpoint/' + model + '/current_model/'
    best_model_path = './checkpoint/' + model + '/best_model/'
    logs = './logs/'
    
    real_label_path, fake_label_path, val_label_path, test_label_path, metrics = get_label_path(protocol)
    
    evaluate_model_path = ''
    
config = DefaultConfigs()
