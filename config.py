class Config:
    image_size = 224
    max_text_len = 40

    batch_size = 16
    num_epochs = 10

    lr_backbone = 1e-5
    lr_head = 1e-4

    num_workers = 2