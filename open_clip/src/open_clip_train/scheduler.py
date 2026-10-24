import numpy as np


def assign_learning_rate_custom(optimizer, updated_lr_dict):
    for param_group in optimizer.param_groups:
        if param_group["name"] in updated_lr_dict.keys():
            param_group["lr"] = updated_lr_dict[param_group["name"]]


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def cosine_lr_custom(optimizer, base_lr_dict, warmup_length, steps):
    updated_lr_dict= {}
    def _lr_adjuster(step):
        for key, base_lr in base_lr_dict.items():
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            updated_lr_dict[key] = lr
        assign_learning_rate_custom(optimizer, updated_lr_dict)
        return updated_lr_dict
    return _lr_adjuster