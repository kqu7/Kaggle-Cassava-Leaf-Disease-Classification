class CFG:
    # -----------
    # Environment
    # ----------- 
    device = 'cpu'  # Options: ['cuda', 'cpu']

    # -----
    # Paths
    # -----
    train_data_path = '/Users/q7/Documents/Cassava/data/'
    train_img_path = '/Users/q7/Documents/Cassava/imgs/train/'
    test_data_path = '/Users/q7/Documents/Cassava/data/'
    test_img_path = '/Users/q7/Documents/Cassava/imgs/test/'
    model_save_path = '/Users/q7/Documents/Cassava/model/'
    log_path = '/Users/q7/Documents/Cassava/log/'
    output_path='/Users/q7/Documents/Cassava/output/'

    # ----
    # Data
    # ----
    n_classes = 5  # Indicates the number of classes for this classification task
    img_size = 384  # Options: [384x384, 512x512]; if VIT or deit is chosen as model, need 384 x 384
    n_epochs = 5  # Indicates the number of epochs trained
    n_folds = 5  # Indicates the number of k-cross validation
    num_workers = 2
    train_batch_size = 4  # Recommended: effnet=16; resnext=8; vit=4; deit=4
    valid_batch_size = 4
    test_batch_size = 3
    train_augmentation_type = 'heavy'
    valid_agumentation_type = 'light'
    use_cutmix = False
    use_2019 = False

    # -----
    # Model
    # -----
    model_name = 'tf_efficientnet_b4_ns' # ['deit_base_patch16_384','vit_large_patch16_384',
                                         # 'tf_efficientnet_b4_ns','resnext50_32x4d']
    pretrained = True  # Controls whether to use the pretrained model
    save_model = False # Controls whether to save the model
    freeze = False  # Controls whether finetune the model  

    # ----------------
    # Pretrained Model
    # ----------------
    pretrained_model_path = '/Users/q7/Documents/Cassava/pretrained-model/'
    model_list = [0, 1, 2, 3] # Specify which ensemble of models to use when inferening

    # --------
    # Training
    # --------
    scheduler_name = 'LambdaLR' # Controls which scheduler to use
                                
    scheduler_update = 'epoch'
    criterion_name = 'CrossEntropyLoss' # Controls which loss function to use
                                        
    optimizer_name = 'AdamW' # Controls which optimizer to use
                             
    lr = 1e-3
    min_lr = 1e-6
    momentum = 0.9
    weight_decay=1e-6 
    T1 = 0.2
    T2 = 1.1
    LABEL_SMOOTH = 0.20
    T_0 = 7
    eps=1e-8

    # -----
    # Utils
    # -----
    mode = "Inference" # Controls whether to train or infer
                      # Options: ["Training", "Inference"]
    print_model = False  # Controls whether to print out the layers of models
    debug = True  # Controls whether to enter debug mode where there are just 100 data 
    use_TTA = True  # Controls whether to use Test Time Augmentation during inference
    seed = 42

# Enter the debug mode
if CFG.debug == True:
    n_epochs = 1
    model_list = [0]
    use_2019 = False

if CFG.use_TTA == True:
    test_batch_size = 1

# If we use deit_base_patch16_384 or vit_large_patch16_384, the image
# size has to be 384x384
if CFG.model_name=='deit_base_patch16_384' or CFG.model_name=='vit_large_patch16_384':
    assert CFG.img_size == 384