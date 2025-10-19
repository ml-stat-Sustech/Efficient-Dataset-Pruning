import torch
import timm


# return classifier with checkpoint
def get_cclf(arch, num_classes, ckpt_path):

    if arch in ['resnet18', 'resnet50']:
        model = timm.create_model(arch, num_classes=num_classes, checkpoint_path=ckpt_path)
    elif arch in ['vit_small', 'vit_base']:
        model = timm.create_model('%s_patch16_224' % arch, num_classes=num_classes, checkpoint_path=ckpt_path)
    else:
        raise Exception('ARCH NOT SUPPORTED')
    return model


# return pretrained classifier
def get_pclf(arch, pretrain, num_classes=1000):

    # 'hybrid'
    if pretrain not in ['weakly', 'fully']:
        raise Exception('PRETRAIN NOT SUPPORTED')
    
    if arch in ['resnet18', 'resnet50']:
        if pretrain == 'weakly':
            model = timm.create_model('%s.fb_swsl_ig1b_ft_in1k' % arch, pretrained=True, num_classes=num_classes)
        else:
            model = timm.create_model('%s.tv_in1k' % arch, pretrained=True, num_classes=num_classes)
    
    elif arch in ['vit_small', 'vit_base']:
        if pretrain == 'weakly':
            # vit_tiny_patch16_224.augreg_in21k
            model = timm.create_model('%s_patch16_224.augreg_in21k' % arch, pretrained=True, num_classes=num_classes)
        else:
            model = timm.create_model('%s_patch16_224.augreg_in1k' % arch, pretrained=True, num_classes=num_classes)

    return model


def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (1, k, d, d)
    """
    res = torch.zeros(mat_a.shape).to(mat_a.device)
    
    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)
    
    return res


def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)