from torch.utils import data
from oarsigrading.dataset.utils import read_gs


class OARSIGradingDataset(data.Dataset):
    def __init__(self, metadata, img_trfs):
        super(OARSIGradingDataset, self).__init__()

        self.meta = metadata
        self.trf = img_trfs

    def __getitem__(self, idx):
        entry = self.meta.iloc[idx]
        img = read_gs(entry.fname)
        #img_res, grades, att_masks = self.trf((img, entry))
        img_res, grades = self.trf((img, entry))
        #return {'img': img_res, 'target': grades.long(), 'att_masks': att_masks.float(), 'ID': entry.ID,
        #        'SIDE': entry.SIDE, 'VISIT': entry.VISIT}
        return {'img': img_res, 'target': grades.long(), 'ID': entry.ID,
                'SIDE': entry.SIDE, 'VISIT': entry.VISIT}

    def __len__(self):
        return self.meta.shape[0]
