import os
import random
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
from scipy import ndimage
from autoattack import AutoAttack

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torch.utils.data import TensorDataset
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader, random_split
import torchvision.models as models


class anti_adversary_wrapper(nn.Module):
    def __init__(self, model, mean=None, std=None, k=0, alpha=1):
        super(anti_adversary_wrapper, self).__init__()
        self.model = model
        self.normalize = False
        self.k = k
        self.alpha = alpha
        # self.loss = nn.CrossEntropyLoss()
        if mean is not None:
            assert std is not None
            self.normalize = True
            # std
            std = torch.tensor(std).view(1, 3, 1, 1)
            self.std = nn.Parameter(std, requires_grad=False).to("cuda")
            # mean
            mean = torch.tensor(mean).view(1, 3, 1, 1)
            self.mean = nn.Parameter(mean, requires_grad=False).to("cuda")
    
    def get_anti_adversary(self, x):
        sudo_label = self.model(x).max(1)[1]
        with torch.enable_grad():#because usually people disables gradients in evaluations
            #This could be changed to randn, but lets see
            anti_adv = torch.zeros_like(x, requires_grad=True)
            for _ in range(self.k):
                loss = F.cross_entropy(self.model(x+anti_adv), sudo_label)
                grad = torch.autograd.grad(loss, anti_adv)
                anti_adv.data -= self.alpha*grad[0].sign()
        return anti_adv
    
    def forward(self, x):#Adaptive update of the anti adversary
        if self.normalize:
            x = (x - self.mean) / self.std
        if self.k > 0 and self.alpha > 0:
            anti_adv = self.get_anti_adversary(x)
            return self.model(x+anti_adv)
        return self.model(x)  

def get_data_utils(dataset_name, batch_size, chunks, num_chunk):
    if dataset_name == 'imagenet':
        from torchvision import transforms
        path = '/local/reference/CV/ILSVR/classification-localization/data/jpeg/val'
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        dataset = ImageFolder(path, transform)
    else:
        dataset_fun = CIFAR10 if dataset_name == 'cifar10' else CIFAR100
        dataset = dataset_fun(root='./data', train=False, download=True,
                            transform=Compose([ToTensor()]))
    tot_instances = len(dataset)
    print('lols', tot_instances)
    assert 1 <= num_chunk <= chunks
    assert tot_instances % chunks == 0
    # inds of current chunk
    inds = np.linspace(0, tot_instances, chunks+1, dtype=int)
    start_ind, end_ind = inds[num_chunk-1], inds[num_chunk]
    # extract data and put in new dataset
    data = [dataset[i] for i in range(start_ind, end_ind)]
    imgs = torch.cat([x.unsqueeze(0) for (x, y) in data], 0)
    labels = torch.cat([torch.tensor(y).unsqueeze(0) for (x, y) in data], 0)
    testset = TensorDataset(imgs, labels)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, 
                            num_workers=2, pin_memory=True, drop_last=False)

    return testloader, start_ind, end_ind


def get_clean_acc(model, testloader, device):
    model.eval()
    n, total_acc = 0, 0
    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            total_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    
    acc = 100. * total_acc / n
    print(f'Clean accuracy: {acc:.4f}')
    return acc


def get_adversary(model, cheap, seed, eps):
    model.eval()
    adversary = AutoAttack(model.forward, norm='Linf', eps=eps, verbose=False)
    adversary.seed = seed
    return adversary

def compute_advs(model, testloader, device, batch_size, cheap, seed, eps):
    model.eval()
    adversary = get_adversary(model, cheap, seed, eps)
    imgs = torch.cat([x for (x, y) in testloader], 0)
    labs = torch.cat([y for (x, y) in testloader], 0)
    advs = adversary.run_standard_evaluation_individual(imgs, labs, 
                                                        bs=batch_size)
    return advs, labs

def compute_adv_accs(model, advs, labels, device, batch_size):
    accs = {}
    all_preds = []
    for attack_name, curr_advs in advs.items():
        dataset = TensorDataset(curr_advs, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=1, pin_memory=True, drop_last=False)
        total_corr = 0
        curr_preds = []
        with torch.no_grad():
            for img, lab in dataloader:
                img, lab = img.to(device), lab.to(device)
                output = model(img)
                pred = output.max(1)[1]
                curr_preds.append(pred)
                total_corr += (pred == lab).sum().item()

        curr_preds = torch.cat(curr_preds)
        all_preds.append(curr_preds)
            
        curr_acc = 100. * total_corr / labels.size(0)
        accs.update({ attack_name : curr_acc })
    
    # compute worst case for each image
    all_preds = torch.cat([x.unsqueeze(0) for x in all_preds])
    temp_labels = labels.unsqueeze(0).expand(len(advs), -1).to(device)
    where_all_correct = torch.prod(all_preds==temp_labels, dim=0) # logical AND
    worst_acc = 100. * where_all_correct.sum().item() / labels.size(0)
    accs.update({ 'rob acc' : worst_acc })

    return accs


def print_to_log(text, txt_file_path):
    with open(txt_file_path, 'a') as text_file:
        print(text, file=text_file)


def print_training_params(args, txt_file_path):
    d = vars(args)
    text = ' | '.join([str(key) + ': ' + str(d[key]) for key in d])
    # Print to log and console
    print_to_log(text, txt_file_path)
    print(text)


def get_model(experiment, k, alpha, dataset):
    if experiment == 'awp': # Adversarial Weight Perturbation
        from experiments.adv_weight_pert import get_model
        model = get_model(k, alpha)
    elif experiment == 'imagenet_pretraining': # ImageNet preatraining
        from experiments.imagenet_pretraining import get_imagenet_pretrained_model
        model = get_imagenet_pretrained_model(k, alpha, dataset)
    model.to("cuda")
    return model


def save_results(advs, labels, accs, args, num_chunk, start_ind, end_ind):
    filename = f'chunk{num_chunk}of{args.chunks}_{start_ind}to{end_ind}'
    # Save adversaries to file
    data_file = osp.join(args.adv_dir, f'advs_{filename}.pth')
    data = {'advs' : advs, 'labels' : labels} # advs is a dict
    torch.save(data, data_file)
    # Log stuff
    log_file = osp.join(args.logs_dir, f'results_{filename}.txt')
    info = '\n'.join([f'{k}:{v}' if k == 'n_instances' else f'{k}:{v:4.2f}'
        for k, v in accs.items()])
    print_to_log(info, log_file)
    print('==> Accuracies: \n', info)

    print(f'Evaluation for chunk {num_chunk} out of {args.chunks} finished.\n'
          f'==> Adversaries saved to {data_file}.\n'
          f'==> Log file saved to {log_file}.\n'
          + 50 * '-' + '\n')
    
    return log_file


def eval_chunk(model, dataset, batch_size, chunks, num_chunk, device, args):
    testloader, start_ind, end_ind = get_data_utils(dataset, batch_size, chunks, 
                                                    num_chunk)
    # Clean acc
    clean_acc = get_clean_acc(model, testloader, device)
    # Compute adversarial instances
    advs, labels = compute_advs(model, testloader, device, batch_size, 
                                args.cheap, args.seed, args.eps)
    # Compute robustness
    accs = compute_adv_accs(model, advs, labels, device, batch_size)
    # Send everything to file
    accs.update({'clean' : clean_acc , 'n_instances' : len(testloader.dataset)})
    log_file = save_results(advs, labels, accs, args, num_chunk, start_ind,
                            end_ind)

    return log_file


def eval_files(log_files, final_log):
    print(f'Evaluating based on these {len(log_files)} files: ', log_files)
    tot_instances = 0
    tot_corr = {}
    for log_file in log_files:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines]
        data = { l.split(':')[0] : float(l.split(':')[1]) for l in lines }
        instances = int(data.pop('n_instances'))
        tot_instances += instances
        for atck, acc in data.items():
            corr = acc * instances
            if atck in tot_corr:
                tot_corr[atck] += corr
            else:
                tot_corr[atck] = corr

    accs = {atck : float(corr)/tot_instances for atck, corr in tot_corr.items()}
    accs.update({ 'n_instances' : tot_instances })
    info = '\n'.join([f'{k}:{v}' if k == 'n_instances' else f'{k}:{v:4.2f}'
        for k, v in accs.items()])
    print_to_log(info, final_log)
    print(f'Saved all results to {final_log}')
