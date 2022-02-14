import torch


def build_optimizer(opt, model):
    ve_params = list(map(id, model.visual_extractor.parameters()))
    ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
    optimizer = getattr(torch.optim, opt.optim)(
        [{'params': model.visual_extractor.parameters(), 'lr': opt.lr_ve},
         {'params': ed_params, 'lr': opt.lr_ed}],
        weight_decay=opt.weight_decay,
        amsgrad=opt.amsgrad
    )
    return optimizer


def build_lr_scheduler(args, optimizer):
    lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)
    return lr_scheduler
