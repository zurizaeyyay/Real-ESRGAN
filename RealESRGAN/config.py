HF_MODELS = {
    2: dict(
        repo_id='ai-forever/Real-ESRGAN',
        filename='RealESRGAN_x2.pth',
    ),
    4: dict(
        repo_id='ai-forever/Real-ESRGAN',
        filename='RealESRGAN_x4.pth',
    ),
    8: dict(
        repo_id='ai-forever/Real-ESRGAN',
        filename='RealESRGAN_x8.pth',
    ),
}


train_config = {
    'scale': 4,
    'num_feat': 64,
    'num_block': 23,
    'num_grow_ch': 32,
    'learning_rate': 2e-4,
    'lr_milestones': [50000, 100000, 200000, 300000],
    'lr_gamma': 0.5,
    'batch_size': 16,
    'patch_size': 192,
    'num_epochs': 500,
    'save_freq': 10,
    'sample_freq': 5,
    'checkpoint_dir': 'checkpoints',
    'weights_dir': 'weights',
    'sample_dir': 'samples'
}


train_config_minimal = {
    'scale': 4,
    'num_feat': 32,      # Smaller model
    'num_block': 12,     # Fewer blocks
    'num_grow_ch': 16,   # Smaller growth
    'learning_rate': 2e-4,
    'batch_size': 4,     # Smaller batch
    'patch_size': 128,   # Smaller patches
    'num_epochs': 10,    # Just a few epochs
    'save_freq': 2,
    'sample_freq': 1,
    'checkpoint_dir': 'checkpoints',
    'weights_dir': 'weights',
    'sample_dir': 'samples'
}