import torch
import solt
import pytorch_lightning as pl
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
import hydra
import numpy as np
from tqdm import tqdm

from oarsigrading.data.utils import build_dataset_meta
from oarsigrading.data.utils import make_weights_for_multiclass, WeightedRandomSampler
from oarsigrading.data.dataset import OARSIGradingDataset
from oarsigrading.training.model import gen_model_parts
from oarsigrading.training.loss import MultiTaskClassificationLoss
from oarsigrading.evaluation.metrics import compute_metrics

from oarsigrading.common import Logger


class OARSIGradingPipeline(pl.LightningModule):
    def __init__(self, cfg, original_cwd=None):
        super(OARSIGradingPipeline, self).__init__()
        self.cfg = cfg

        self.oai_meta = None
        self.most_meta = None

        self.cv_splits = None

        self.train_metadata = None
        self.val_metadata = None
        self.test_metadata = None
        self.mean_std = None

        if original_cwd is None:
            original_cwd = Path(hydra.utils.get_original_cwd())
        self.workdir = original_cwd / cfg.workdir
        self.dataset_dir = original_cwd / cfg.data.dataset_dir

        self.features, self.classifier = gen_model_parts(cfg)
        self.criterion = MultiTaskClassificationLoss()

    def prepare_data(self) -> None:
        if not (self.workdir / 'oai_meta.pkl').is_file():
            oai_meta, most_meta = build_dataset_meta(self.dataset_dir)
            oai_meta.to_pickle(self.workdir / 'oai_meta.pkl', compression='infer', protocol=4)
            most_meta.to_pickle(self.workdir / 'most_meta.pkl', compression='infer', protocol=4)
        else:
            oai_meta = pd.read_pickle(self.workdir / 'oai_meta.pkl')
            most_meta = pd.read_pickle(self.workdir / 'most_meta.pkl')

        # Removing the corner cases
        most_meta = most_meta[(most_meta.XRKL >= 0) & (most_meta.XRKL <= 4)]
        oai_meta = oai_meta[(oai_meta.XRKL >= 0) & (oai_meta.XRKL <= 4)]

        gkf = GroupKFold(self.cfg.data.n_folds)
        train_meta_all = oai_meta if self.cfg.data.train_on == 'oai' else most_meta
        self.cv_splits = [x for x in gkf.split(X=train_meta_all,
                                               y=train_meta_all.XRKL.values,
                                               groups=train_meta_all['ID'].values)]

        train_idx, val_idx = self.cv_splits[self.cfg.data.fold_id]
        # The metadata is setup
        self.train_metadata = train_meta_all.iloc[train_idx]
        self.val_metadata = train_meta_all.iloc[val_idx]
        self.test_metadata = most_meta if self.cfg.data.train_on == 'oai' else oai_meta

        self.mean_std = self.init_mean_std()

    def init_mean_std(self):
        if (self.workdir / f'mean_std_{self.cfg.data.train_on}.npy').is_file():
            tmp = np.load(self.workdir / f'mean_std_{self.cfg.data.train_on}.npy')
            mean_vector, std_vector = tmp
        else:
            train_trfs = solt.utils.from_yaml(self.cfg.data.trfs.train.to_container(resolve=True))

            train_ds = OARSIGradingDataset(self.cfg,
                                           self.train_metadata,
                                           train_trfs, normalize=False)
            tmp_loader = DataLoader(train_ds, batch_size=self.cfg.training.batch_size,
                                    num_workers=self.cfg.training.n_threads)
            mean_vector = None
            std_vector = None
            for batch in tqdm(tmp_loader, total=len(tmp_loader)):
                imgs = batch['image']
                if mean_vector is None:
                    mean_vector = np.zeros(imgs.size(1))
                    std_vector = np.zeros(imgs.size(1))
                for j in range(mean_vector.shape[0]):
                    mean_vector[j] += imgs[:, j, :, :].mean()
                    std_vector[j] += imgs[:, j, :, :].std()

            mean_vector /= len(tmp_loader)
            std_vector /= len(tmp_loader)
            np.save(self.workdir / f'mean_std_{self.cfg.data.train_on}.npy',
                    [mean_vector.astype(np.float32), std_vector.astype(np.float32)])

        return mean_vector, std_vector

    def init_sampler(self):
        if self.cfg.data.sampling == 'klw':
            _, weights = make_weights_for_multiclass(self.train_metadata.XRKL.values.astype(int))
        elif self.cfg.data.sampling == 'mtw':
            cols = ['XROSTL', 'XROSFL', 'XRJSL', 'XROSTM', 'XROSFM', 'XRJSM']
            weights = torch.stack([make_weights_for_multiclass(self.train_metadata[col].values.astype(int))[1]
                                   for col in cols], 1).max(1)[0]
        else:
            weights = None

        if weights is not None:
            sampler = WeightedRandomSampler(weights, self.train_metadata.shape[0], True)
        else:
            sampler = None
        return sampler

    def train_dataloader(self):
        train_trfs = solt.utils.from_yaml(self.cfg.data.trfs.train.to_container(resolve=True))

        train_ds = OARSIGradingDataset(self.cfg,
                                       self.train_metadata,
                                       train_trfs,
                                       mean=self.mean_std[0],
                                       std=self.mean_std[1])
        sampler = self.init_sampler()
        train_loader = DataLoader(train_ds, batch_size=self.cfg.training.batch_size,
                                  num_workers=self.cfg.training.n_threads,
                                  drop_last=True, sampler=sampler)
        return train_loader

    def val_dataloader(self):
        val_dataset = OARSIGradingDataset(self.cfg,
                                          self.val_metadata,
                                          solt.utils.from_yaml(self.cfg.data.trfs.eval.to_container(resolve=True)),
                                          mean=self.mean_std[0],
                                          std=self.mean_std[1])

        val_loader = DataLoader(val_dataset, batch_size=self.cfg.training.val_batch_size,
                                num_workers=self.cfg.training.n_threads, shuffle=False)
        return val_loader

    def configure_optimizers(self):
        if self.cfg.training.unfreeze_epoch == 0:
            self.classifier.train(True)
            self.features.train(True)
            p = [{'params': self.features.parameters()},
                 {'params': self.classifier.parameters()}]
        else:
            self.classifier.train(True)
            self.features.train(False)

            for name, param in self.features.named_parameters():
                param.requires_grad = False
            p = [{'params': self.classifier.parameters()}]
        return torch.optim.Adam(params=p,
                                lr=self.cfg.training.optimizer.lr,
                                weight_decay=self.cfg.training.optimizer.weight_decay)

    def forward(self, x):
        return self.classifier(self.features(x))

    def training_step(self, batch, batch_idx):
        preds = self(batch['image'])
        loss = self.criterion(preds, batch['target'].squeeze())
        to_logs = {'loss/train': loss}
        return {'loss': loss, 'log': to_logs}

    def validation_step(self, batch, batch_idx):
        targets = batch['target'].squeeze()
        preds = self(batch['image'])
        loss = self.criterion(preds, targets)
        outputs = {'loss': loss, 'preds': [p.softmax(1) for p in preds], 'targets': targets}
        return outputs

    def validation_epoch_end(self, outputs):
        val_loss = 0
        preds = []
        targets = []
        for batch in outputs:
            val_loss += torch.mean(batch['loss'])
            for i, p in enumerate(batch['preds']):
                batch['preds'][i] = p.argmax(1).unsqueeze(0)
            batch['preds'] = torch.cat(batch['preds']).squeeze().T
            preds.append(batch['preds'])
            targets.append(batch['targets'])
        targets = torch.cat(targets, 0)
        preds = torch.cat(preds, 0)
        metrics = {'loss/val': val_loss / len(outputs)}
        metrics.update({f'val/{k}': v for k, v in compute_metrics(targets, preds,
                                                                  no_kl=self.cfg.training.no_kl).items()})

        return {'progress_bar': metrics, 'log': metrics, 'loss': metrics[f'loss/val']}

    def on_epoch_start(self) -> None:
        if self.cfg.training.unfreeze_epoch == self.current_epoch and self.current_epoch > 0:
            Logger.log_info(f'[{self.current_epoch}] Unfreezing the whole model...')
            for name, param in self.features.named_parameters():
                param.requires_grad = True
            self.features.train(True)
            self.trainer.optimizers[0].add_param_group({'params': self.features.parameters()})

        for milestone in self.cfg.training.lr_scheduler.milestones:
            if self.current_epoch == milestone:
                for opt_id, optim in enumerate(self.trainer.optimizers):
                    for pg_id, param_group in enumerate(optim.param_groups):
                        new_lr = param_group["lr"] * self.cfg.training.lr_scheduler.gamma
                        msg = f'[{self.current_epoch}] Dropping LR from {param_group["lr"]} to {new_lr} in ' \
                              f'optimizer {opt_id}, param group {pg_id}'
                        Logger.log_info(msg)
                        param_group["lr"] = new_lr
