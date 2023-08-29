import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils
from matplotlib import pyplot as plt
from torchvision import models as torch_models
import sys
import time
from datetime import datetime
from utils import SingleChannelModel
from torch.autograd import Variable
from paper_models import *
import CIFAR100_models
import TinyImageNet_models

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_class_dict = {
    "inception_v3": (torch_models.inception_v3, 299),
    "resnet50": (torch_models.resnet50, 224),
    "vgg16_bn": (torch_models.vgg16_bn, 224),
    "vgg16": (torch_models.vgg16, 224),
                    }

class PretrainedModel():
    def __init__(self, modelname):
        model_pt = model_class_dict[modelname][0](pretrained=True)
        self.model = nn.DataParallel(model_pt.to(device))
        self.model.eval()
        self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1).to(device)
        self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1).to(device)

    def predict(self, x):
        out = (x - self.mu) / self.sigma
        return self.model(out)

    def forward(self, x):
        out = (x - self.mu) / self.sigma
        return self.model(out)

    def __call__(self, x):
        return self.predict(x)


def random_target_classes(y_pred, n_classes):
    y = torch.zeros_like(y_pred)
    for counter in range(y_pred.shape[0]):
        l = list(range(n_classes))
        l.remove(y_pred[counter])
        t = torch.randint(0, len(l), size=[1])
        y[counter] = l[t] + 0

    return y.long()


# def save_adv(adv, n_batches, bs):
#     for counter in range(adv.shape[0]):
#         torchvision.utils.save_image(adv[counter], './adv_imgs_fix_side/' + str(n_batches * bs + counter) + '.png',
#                                      normalize=False)


def show_image(x0):
    tmp = x0.clone()
    tmp = tmp.transpose(0, 1)
    tmp = tmp.transpose(1, 2)
    tmp = tmp.cpu()
    plt.imshow(tmp)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='ImageNet')
    parser.add_argument('--model', default='inception_v3', type=str)
    parser.add_argument('--loss', type=str, default='ce')   ## 'margin loss or cross-entropy loss'
    parser.add_argument('--bs', type=int, default=100)
    parser.add_argument('--n_queries', type=int, default=2000)
    parser.add_argument('--targeted', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./results')

    # Sparse-RS parameter
    parser.add_argument('--miu_init', type=float, default=.4)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--width', type=int, default=1)
    parser.add_argument('--length', type=int, default=32)

    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    

    if args.dataset == 'ImageNet':
        class_number = 1000
        # load pretrained model
        model = PretrainedModel(args.model)
        assert not model.model.training

        # load data
        IMAGENET_SL = model_class_dict[args.model][1]
        IMAGENET_PATH = "/data2/ranyu/ImageNet_val"
        imagenet = datasets.ImageFolder(IMAGENET_PATH,
                                        transforms.Compose([
                                            transforms.Resize(IMAGENET_SL),
                                            transforms.CenterCrop(IMAGENET_SL),
                                            transforms.ToTensor()
                                        ]))
       
        # randomly pick 500 images
        test_loader = data.DataLoader(imagenet, batch_size=500, shuffle=True, num_workers=0)
    
    elif args.dataset == 'TinyImageNet':
        class_number = 200
        # load pretrained model
        model = TinyImageNet_models.__dict__[args.model]()
        model.eval()
        model.to(device)
        filename = './models/TinyImageNet_'+args.model+'.pth'
        model.load_state_dict(torch.load(filename)['state_dict'])
    
        # load data
        TINYIMAGENET_PATH = '/data2/ranyu/tiny-imagenet-200/val/images'
        test_loader = load_TinyImageNet_data(data_root=TINYIMAGENET_PATH, test_batch_size=500)

    elif args.dataset == 'CIFAR100':
        class_number = 100
        # load pretrained model
        if args.model == 'resnet50':
            # resnet backbone
            model = CIFAR100_models.__dict__['resnet50'](num_classes=100)
            model.eval()
            model.to(device)
            filename = './models/cifar100_res.pth'
            model.load_state_dict(torch.load(filename)['state_dict'])
        elif args.model == 'desnet121':
            model = CIFAR100_models.__dict__['densenet121'](num_classes=100)
            model.eval()
            model.to(device)
            filename = './models/cifar100_densenet121.pth'
            model.load_state_dict(torch.load(filename)['state_dict'])
        else:
            print('Do not support.')
            exit(0)
        # load data
        test_loader = load_cifar100_data(test_batch_size=500)
    
    elif args.dataset == 'CIFAR10':
        class_number = 10
        # load pretrained model
        if args.model == 'cnn':
            # resnet backbone
            model = CIFAR10_CNN()
            model.eval()
            model.to(device)
            load_model(model, './models/cifar10_cnn.pth')
        elif args.model == 'vgg16':
            # vgg backbone
            model = VGG_plain('VGG16', 10)
            model.eval()
            model.to(device)
            load_model(model, './models/cifar10_vgg16.pth')
        else:
            print('Do not support.')
            exit(0)

        # load data
        test_loader = load_cifar10_data(test_batch_size=500)


    else: 
        print('Do not support.')
        exit(0)

    testiter = iter(test_loader)
    x_test, y_test = next(testiter) 

    # log directory 
    logsdir = '{}/logs_CSPA'.format(args.save_dir)
    savedir = '{}/results_CSPA'.format(args.save_dir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    if not os.path.exists(logsdir):
        os.makedirs(logsdir)

    param_run = 'seed_{:.0f}_{}_bs_{}_nqueries_{:.0f}_miu_init_{:.2f}_loss_{}_targeted_{}_interval_{:.0f}_width_{:.0f}_length_{:.0f}'.format(
         args.seed, args.model, args.bs, args.n_queries, args.miu_init, args.loss, args.targeted, args.interval, args.width, args.length)


    from CSPAttack import CSPA

    adversary = CSPA(model, n_queries=args.n_queries,
                         miu_init=args.miu_init,
                         log_path='{}/log_run_{}_{}.txt'.format(logsdir, str(datetime.now())[:-7], param_run),
                         loss=args.loss, 
                         targeted=args.targeted, 
                         seed=args.seed,
                         interval=args.interval, 
                         width=args.width,    
                         length=args.length,
                         device=device)

    # set target classes
    if args.targeted:
        y_tar = random_target_classes(y_test, class_number)
        print('target labels', y_tar)
       

    bs = min(args.bs, 500)
    adv_complete = x_test.clone()
    qr_complete = torch.zeros([x_test.shape[0]]).cpu()
    pred = torch.zeros([0]).float().cpu()
    with torch.no_grad():
        # find points originally correctly classified
        for counter in range(x_test.shape[0] // bs):
            x_curr = x_test[counter * bs:(counter + 1) * bs].to(device)
            y_curr = y_test[counter * bs:(counter + 1) * bs].to(device)
            output = model(x_curr)
            pred = torch.cat((pred, (output.max(1)[1] == y_curr).float().cpu()), dim=0)
            

        adversary.logger.log('clean accuracy {:.2%}'.format(pred.mean()))

        n_batches = pred.sum() // bs + 1 if pred.sum() % bs != 0 else pred.sum() // bs
        n_batches = n_batches.long().item()
        ind_to_fool = (pred == 1).nonzero().squeeze()

        # run the attack
        pred_adv = pred.clone()
        for counter in range(n_batches):
            x_curr = x_test[ind_to_fool[counter * bs:(counter + 1) * bs]].to(device)
            if not args.targeted:
                y_curr = y_test[ind_to_fool[counter * bs:(counter + 1) * bs]].to(device)
            else:
                y_curr = y_tar[ind_to_fool[counter * bs:(counter + 1) * bs]].to(device)
            qr_curr, adv = adversary.attack(x_curr, y_curr)

            output = model(adv.to(device))
            if not args.targeted:
                acc_curr = (output.max(1)[1] == y_curr).float().cpu()
            else:
                acc_curr = (output.max(1)[1] != y_curr).float().cpu()
            pred_adv[ind_to_fool[counter * bs:(counter + 1) * bs]] = acc_curr.clone()
            adv_complete[ind_to_fool[counter * bs:(counter + 1) * bs]] = adv.cpu().clone()
            qr_complete[ind_to_fool[counter * bs:(counter + 1) * bs]] = qr_curr.cpu().clone()

            adversary.logger.log('batch {}/{} - {:.0f} of {} successfully perturbed'.format(
                counter + 1, n_batches, x_curr.shape[0] - acc_curr.sum(), x_curr.shape[0]))

            # save_adv(adv, counter, bs)



        # statistics
        res = (adv_complete - x_test != 0.).max(dim=1)[0].sum(dim=(1, 2))
        adversary.logger.log(
            'max L0 perturbation {:.0f} - nan in img {} - max img {:.5f} - min img {:.5f}'.format(
                 res.max(), (adv_complete != adv_complete).sum(), adv_complete.max(), adv_complete.min()))

        ind_corrcl = pred == 1.
        ind_succ = (pred_adv == 0.) * (pred == 1.)

        str_stats = 'success rate={:.0f}/{:.0f} ({:.2%}) \n'.format(
            pred.sum() - pred_adv.sum(), pred.sum(), (pred.sum() - pred_adv.sum()).float() / pred.sum())
        qr_complete[~ind_succ] = args.n_queries + 0
        str_stats += '[correctly classified points] avg # queries {:.2f} - med # queries {:.2f}\n'.format(
            qr_complete[ind_corrcl].float().mean(), torch.median(qr_complete[ind_corrcl].float()))
        adversary.logger.log(str_stats)

        # save results
       
       
        torch.save({'adv': adv_complete, 'qr': qr_complete},'{}/{}.pth'.format(savedir, param_run))




