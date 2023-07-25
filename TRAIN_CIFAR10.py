import torch
import torchvision
from torchvision import transforms
from torch.utils.data import TensorDataset
import argparse
from models.resnet import *
from models.ViT import ViT
from pgd_attack import eval_adv_test_whitebox
import wandb
import numpy as np
import random
import yaml
import os


parser = argparse.ArgumentParser(description='Do PAG Imply Robustness?')
parser.add_argument('--config_path', type=str, default='./configs/cifar10_sbg_rn18.yaml', help='training config path')
args, _ = parser.parse_known_args()

with open(args.config_path, "r") as f:
    config = yaml.safe_load(f)

output_dir = config["chekpoint_folder"]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Reproducibility
random.seed(config["seed"])
torch.manual_seed(config["seed"])
torch.cuda.manual_seed_all(config["seed"])
np.random.seed(config["seed"])


model_name = f'{config["arch"]}-cifar10-grad_source-{config["grad_source"]}-pag_coeff-{config["pag_coeff"]}' \
             f'-subsample-{config["num_grads_per_image"]}-seed-{config["seed"]}'

if config["use_wandb"]:
    wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], config=config)
    wandb.run.name = model_name

# get data - train
data_stats = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
data = torch.load('data/c10_sbg_data.pt')
labels = torch.load('data/c10_sbg_label.pt')

dataset = TensorDataset(data, labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True,
                                         drop_last=True, num_workers=2)
aug = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])

# get data - test
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# get model
if config["arch"] == 'vit':
    model = torch.nn.DataParallel(ViT()).cuda()
else:
    model = torch.nn.DataParallel(ResNet18()).cuda()
# print architecture details
print(f'Using {config["arch"]} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} learnable params')

# get optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"],
                            weight_decay=config["weight_decay"])


# train function
def train_func(model, train_loader, optimizer):
    model.train()
    #
    for batch_idx, (data, target) in enumerate(train_loader):
        images, target = data.cuda(), target.cuda()
        # split data to images and "ground-truth" PAG
        images, pag = images[:, 0], images[:, 1:]
        optimizer.zero_grad()

        pred = model(aug(images))
        # CE loss
        loss_ce = torch.nn.CrossEntropyLoss()(pred, target)
        # PAG loss
        pag_loss = 0
        if pag_coeff != 0 and config["num_grads_per_image"] > 0:
            # create a (batch_size * num_grads_per_image) batch to enable a single pass in the network
            ims = []
            pag_new = []
            target_new = []
            for rep in range(config["batch_size"]):
                # for each image in the batch, select #num_grads_per_image target classes for pag loss
                targets = torch.randperm(10, device=target.device)[:config["num_grads_per_image"]]
                images_i = images[rep].repeat(config["num_grads_per_image"], 1, 1, 1)    # repeat the image
                pag_i = torch.index_select(pag[rep].squeeze(0), 0, targets)   # select the PAG gt based on targets
                ims.append(images_i)
                pag_new.append(pag_i)
                target_new.append(targets)
            ims = torch.cat(ims).requires_grad_(True)
            pag_new = torch.cat(pag_new)   # (sub)set of ground-truth PAG
            target_new = torch.cat(target_new)  # target classes for PAG-loss calculations

            dummy_loss = torch.nn.CrossEntropyLoss()(model(aug(ims)), target_new)
            # create graph to enable input-gradients loss
            grad, = torch.autograd.grad(-1 * dummy_loss, [ims], create_graph=True)
            pag_loss = 10 * (1. - torch.nn.CosineSimilarity(dim=1)(grad.view(target_new.shape[0], -1),
                                                                   pag_new.view(target_new.shape[0], -1)).mean())

        loss = loss_ce + pag_coeff * pag_loss
        # report training statistics
        if batch_idx % 100 == 0:
            print(f'Train loss in batch {batch_idx}: {loss} | CE loss: {loss_ce} | PAG loss (before | after coeff): '
                  f'{pag_loss} | {pag_coeff * pag_loss}')
        loss.backward()
        optimizer.step()


def adjust_learning_rate(optimizer, epoch):
    """a multistep learning rate drop mechanism"""
    lr = config["lr"]
    epochs = config["epochs"]
    if epoch >= 0.5 * epochs:
        lr = config["lr"] * 0.1
    if epoch >= 0.75 * epochs:
        lr = config["lr"] * 0.01
    if epoch >= epochs:
        lr = config["lr"] * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f'epoch : {epoch} -> lr : {lr}')


def get_pag_coeff(epoch):
    """returns the current pag_coeff"""
    if "step_pag_coeff" in config.keys() and config["step_pag_coeff"] and epoch <= 50:
        return config["pag_coeff"] * 2. / 3.
    return config["pag_coeff"]


for epoch in range(1, config["epochs"] + 1):
    # adjust learning rate for SGD
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adjust_learning_rate(optimizer, epoch)
    # get pag_coeff
    pag_coeff = get_pag_coeff(epoch)
    # training for one epoch
    print(f"Training epoch {epoch}")
    train_func(model, dataloader, optimizer)
    # evaluate robustness on L2, epsilon 0.5 (not AutoAttack)
    c_acc, r_acc = eval_adv_test_whitebox(model=model, device=device, test_loader=testloader, epsilon=0.5, num_steps=7,
                                          step_size=1.5 * 0.5 / 7, random=True, verbose=False, norm='l_2',
                                          stats=data_stats)
    if config["use_wandb"]:
        wandb.log({"c_acc": c_acc, "r_acc": r_acc})
    # save model
    if epoch % config["checkpoint_freq"] == 0 or epoch == config["epochs"]:
        torch.save(model.state_dict(), f'{config["chekpoint_folder"]}/{model_name}-epoch-{epoch}.pt')
