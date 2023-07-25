from __future__ import print_function
import torchvision
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim


def norms(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]


# L2 attack
def _pgd_whitebox_l2(model, X, y, epsilon, num_steps, step_size, random, device, verbose=True,
                     return_x=False, stats=None):
    def normalize_data(x):
        if stats:
            return torchvision.transforms.Normalize(*stats)(x)
        else:
            return x

    out = model(normalize_data(X))
    err = (out.data.max(1)[1] != y.data).float().sum().detach().item()
    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-8./255, 8./255).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(normalize_data(X_pgd)), y)
        loss.backward()
        #
        eta = step_size * X_pgd.grad.detach() / norms(X_pgd.grad.detach())
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = X_pgd.data - X.data
        eta *= epsilon / norms(eta).clamp(min=epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(normalize_data(X_pgd)).data.max(1)[1] != y.data).float().sum().detach().item()
    if verbose:
        print('err pgd (white-box): ', err_pgd)
    if return_x:
        return X_pgd
    return err, err_pgd


# Linf attack
def _pgd_whitebox(model, X, y, epsilon, num_steps, step_size, random, device, verbose=True,
                  return_x=False, stats=None):
    def normalize_data(x):
        if stats:
            return torchvision.transforms.Normalize(*stats)(x)
        else:
            return x
    out = model(normalize_data(X))
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(normalize_data(X_pgd)), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(normalize_data(X_pgd)).data.max(1)[1] != y.data).float().sum()
    if verbose:
        print('err pgd (white-box): ', err_pgd)
    if return_x:
        return X_pgd
    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader, epsilon, num_steps, step_size, random, verbose=True,
                           norm='l_inf', stats=None):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    attack_batch = _pgd_whitebox
    if norm == 'l_2':
        attack_batch = _pgd_whitebox_l2
    num_samples = len(test_loader.sampler)
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = attack_batch(model, X, y, epsilon, num_steps, step_size, random, device,
                                               verbose=verbose, stats=stats)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('================================================================')
    print('clean accuracy: ', (num_samples - natural_err_total) / num_samples)
    print('robust accuracy: ', (num_samples - robust_err_total) / num_samples)
    return (num_samples - natural_err_total) / num_samples, (num_samples - robust_err_total) / num_samples
