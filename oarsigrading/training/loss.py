from torch import nn


class MultiTaskClassificationLoss(nn.Module):
    def __init__(self):
        super(MultiTaskClassificationLoss, self).__init__()
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target_cls):
        loss = 0
        n_tasks = len(pred)

        for task_id in range(n_tasks):
            loss += self.cls_loss(pred[task_id], target_cls[:, task_id])

        loss /= n_tasks

        return loss

