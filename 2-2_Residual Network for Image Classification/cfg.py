

## You could add some configs to perform other training experiments...

LeNet_cfg = {
    'model_type': 'myMode_final_final',#'LeNet',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.95,#0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 32,#16, #batch大一點，算比較快
    'lr':0.01,
    'milestones': [15, 25, 30],#[5, 15, 25, 30],#[15, 25],
    'num_out': 10, #10個classes (不動)
    'num_epoch': 40, #30,
    
}