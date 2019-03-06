import torch
import numpy as np
import torch.nn.functional as F


def five_crop(img, size):
    """Returns a stacked 5 crop
    """
    img = img.clone()
    _, h, w = img.size()
    # get central crop
    c_cr = img[:, h//2-size//2:h//2+size//2, w//2-size//2:w//2+size//2]
    # upper-left crop
    ul_cr = img[:, 0:size, 0:size]
    # upper-right crop
    ur_cr = img[:, 0:size, w-size:w]
    # bottom-left crop
    bl_cr = img[:, h-size:h, 0:size]
    # bottom-right crop
    br_cr = img[:, h-size:h, w-size:w]
    return torch.stack((c_cr, ul_cr, ur_cr, bl_cr, br_cr))


def eval_batch(net, inputs, target):
    """Evaluates TTA for a single batch"""

    inp_lat = None
    inp_med = None
    if not isinstance(inputs, tuple):
        if inputs.device != 'cuda':
            inputs = inputs.to('cuda')
            inputs = inputs.squeeze()
    else:
        inp_med, inp_lat = inputs
        if inp_med.device != 'cuda':
            inp_med = inp_med.to('cuda')

        if inp_lat.device != 'cuda':
            inp_lat = inp_lat.to('cuda')

    target = target.squeeze()

    if not isinstance(inputs, tuple):
        if len(inputs.size()) == 5:
            bs, n_crops, c, h, w = inputs.size()
            outputs = net(inputs.view(-1, c, h, w))
            outputs = [F.softmax(o, 1).view(bs, n_crops, o.size()[-1]).mean(1) for o in outputs]
        else:
            outputs = net(inputs)
    else:
        if len(inp_med.size()) == 5:
            bs, n_crops, c_m, h_m, w_m = inp_med.size()
            _, _, c_l, h_l, w_l = inp_lat.size()
            outputs = net(inp_med.view(-1, c_m, h_m, w_m), inp_lat.view(-1, c_m, h_m, w_m))

            outputs = [F.softmax(o, 1).view(bs, n_crops, o.size()[-1]).mean(1) for o in outputs]
        else:
            outputs = net(inp_med, inp_lat)

    tmp_preds = np.zeros(target.size(), dtype=np.int64)
    for task_id, o in enumerate(outputs):
        tmp_preds[:, task_id] = outputs[task_id].to('cpu').squeeze().argmax(1)

    return tmp_preds
