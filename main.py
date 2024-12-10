import os
import sys
from models.build import build_model
from utils.parser import parse_args, load_config
from utils.log import init_wandb, set_time_to_log_dir
from datasets.build import update_cfg_from_dataset
from trainer import build_trainer
from predictor import Predictor
from utils.misc import set_seeds, set_devices
from utils.regularizer import regularizations

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
def main():
    args = parse_args()
    cfg, cls_gsheet = load_config(args)
    update_cfg_from_dataset(cfg, cfg.DATA.NAME)

    # select cuda devices
    set_devices(cfg.VISIBLE_DEVICES)

    # set wandb logger
    if cfg.WANDB.ENABLE:
        init_wandb(cfg)
    if cfg.LOG_TIME:
        set_time_to_log_dir(cfg)
        
    with open(os.path.join(cfg.RESULT_DIR, 'config.txt'), 'w') as f:
        f.write(cfg.dump())

    # set random seed
    set_seeds(cfg.SEED)

    # build model
    model = build_model(cfg)
    trainer = build_trainer(cfg, model, finetune = False)

    if cfg.TRAIN.ENABLE: 
        trainer.train()
        
    if cfg.TRAIN.FINETUNE.ENABLE:
        model = trainer.load_best_model()
        reg_class = regularizations(model, trainer) if cfg.TRAIN.FINETUNE.REGULARIZATION else None
        trainer = build_trainer(cfg, model, finetune = True)
        trainer.reg_cls = reg_class
        trainer.train()
        
    if cfg.TEST.ENABLE:
        model = trainer.load_best_model()
        
        # Finetune에 따라 바뀌어야 함
        predictor = Predictor(cfg, model, cls_gsheet, finetune = cfg.TRAIN.FINETUNE.ENABLE)
        predictor.predict()
        if cfg.TEST.VIS_ERROR or cfg.TEST.VIS_DATA:
            predictor.visualize()


if __name__ == '__main__':
    main()