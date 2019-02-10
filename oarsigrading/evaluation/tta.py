import torch


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


def eval_batch(net, sample, return_features=False):
    """Evaluates TTA for a single batch"""
    with torch.no_grad():
        inputs = sample['img'].squeeze().to("cuda")
        if len(inputs.size()) == 5:
            bs, n_crops, c, h, w = inputs.size()
            out = net(inputs.view(-1, c, h, w), return_features)
            if return_features:
                out, features = out

            # TODO: refactor for this task
            raise NotImplementedError
            # out = torch.sigmoid(out)
            # out = out.view(bs, n_crops, out.size()[-1]).mean(1)

            # Averaging the features across TTA dimension
            #if return_features:
            #    features = features.view(bs, n_crops, features.size()[-1]).mean(1)
        else:
            out = net(inputs, return_features)
            if return_features:
                out, features = out
            out = torch.sigmoid(out)

    if return_features:
        return out, features
    return out