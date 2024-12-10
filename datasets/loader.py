from torch.utils.data import DataLoader

from datasets.build import build_dataset


def construct_loader(cfg, split):
    if split == "train":
        batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = cfg.TRAIN.SHUFFLE
        drop_last = cfg.TRAIN.DROP_LAST
        finetune = False
    elif split == "train_finetune":
        batch_size = cfg.TRAIN.FINETUNE.BATCH_SIZE
        shuffle = cfg.TRAIN.FINETUNE.SHUFFLE
        drop_last = cfg.TRAIN.FINETUNE.DROP_LAST
        split = "train"
        finetune = True
        
    elif split == "val":
        batch_size = cfg.VAL.BATCH_SIZE
        shuffle = cfg.VAL.SHUFFLE
        drop_last = cfg.VAL.DROP_LAST
        finetune = False
    elif split == "val_finetune":
        batch_size = cfg.VAL.FINETUNE.BATCH_SIZE
        shuffle = cfg.VAL.FINETUNE.SHUFFLE
        drop_last = cfg.VAL.FINETUNE.DROP_LAST
        split = "val"
        finetune = True
        
    elif split == "test":
        batch_size = cfg.TEST.BATCH_SIZE
        shuffle = cfg.TEST.SHUFFLE
        drop_last = cfg.TEST.DROP_LAST
        finetune = True
    else:
        raise ValueError

    dataset = build_dataset(cfg, split, finetune) # Crypto Class 들어있음
    #print(split,len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        prefetch_factor=cfg.DATA_LOADER.PREFETCH_FACTOR,
        persistent_workers=cfg.DATA_LOADER.PERSISTENT_WORKERS
    )

    return loader


def get_train_dataloader(cfg, finetune=False):
    return construct_loader(cfg, "train") if not finetune else construct_loader(cfg, "train_finetune")


def get_val_dataloader(cfg, finetune=False):
    return construct_loader(cfg, "val") if not finetune else construct_loader(cfg, "val_finetune")


def get_test_dataloader(cfg):
    return construct_loader(cfg, "test")