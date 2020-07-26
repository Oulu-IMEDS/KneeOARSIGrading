import hydra
from pathlib import Path
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import Trainer
from oarsigrading.training.pipeline import OARSIGradingPipeline


@hydra.main(config_path=Path.cwd() / 'conf' / 'config.yaml')
def main(cfg):
    seed_everything(cfg.seed)

    lightning_module = OARSIGradingPipeline(cfg)
    trainer = Trainer(replace_sampler_ddp=False,
                      distributed_backend='dp',
                      gpus=1,
                      auto_select_gpus=True,
                      deterministic=True,
                      num_sanity_val_steps=0)
    trainer.fit(lightning_module)


if __name__ == '__main__':
    main()