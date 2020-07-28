import hydra
import torch
from pathlib import Path
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import Trainer
from oarsigrading.training.pipeline import OARSIGradingPipeline


@hydra.main(config_path=Path.cwd() / 'conf' / 'config.yaml')
def main(cfg):
    seed_everything(cfg.seed)

    lightning_module = OARSIGradingPipeline(cfg)

    trainer = Trainer(replace_sampler_ddp=False,
                      gpus=3,
                      distributed_backend='dp',
                      auto_select_gpus=True,
                      deterministic=True,
                      num_sanity_val_steps=0,
                      print_nan_grads=True,
                      max_epochs=cfg.training.n_epochs)
    trainer.fit(lightning_module)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()