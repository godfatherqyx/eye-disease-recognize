from config.default import cfg


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
           specified step
    """
    lr = cfg.LR * (gamma ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr