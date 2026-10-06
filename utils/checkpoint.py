import torch


def trainable_state_dict(model):
    """State dict containing only parameters that require grad.

    Frozen backbone weights are skipped so checkpoints stay small.
    """
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k in trainable}


def save_trainable(model, path):
    torch.save(trainable_state_dict(model), path)


def load_trainable(model, path, device=None):
    state = torch.load(path, map_location=device, weights_only=True)
    return model.load_state_dict(state, strict=False)
