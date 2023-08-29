import torch
import time
import math
import torch.nn.functional as F
import numpy as np
import copy
import sys
import torchvision.utils
from matplotlib import pyplot as plt
from utils import Logger
import os


class CSPA():

    def __init__(
            self,
            model, 
            n_queries,
            miu_init,
            log_path,
            loss, 
            targeted, 
            seed,
            interval, 
            width,    
            length,
            device):
        
        self.predict = model
        self.n_queries = n_queries
        self.miu_init = miu_init
        self.loss = loss 
        self.targeted = targeted 
        self.seed = seed
        self.interval = interval 
        self.width = width
        self.length = length
        self.device = device
        self.logger = Logger(log_path)



    def margin_or_ce(self, x, y):
        """
        :param y:        correct labels if untargeted else target labels
        """
        logits = self.predict(x)
        ce = F.cross_entropy(logits, y, reduction='none')
        u = torch.arange(x.shape[0])
        y_corr = logits[u, y].clone()
        logits[u, y] = -float('inf')
        y_others = logits.max(dim=-1)[0]
        # Untargeted 
        if not self.targeted:
            if self.loss == 'ce':
                return y_corr - y_others, -1. * ce
            elif self.loss == 'margin':
                return y_corr - y_others, y_corr - y_others
        # Targeted 
        else:
            if self.loss == 'ce':
                return y_others - y_corr, ce
            elif self.loss == 'margin':
                return y_others - y_corr, y_others - y_corr


    def random_choice(self, shape):
        t = 2 * torch.rand(shape).to(self.device) - 1
        return torch.sign(t)

    def random_int(self, low=0, high=1, shape=[1]):
        t = low + (high - low) * torch.rand(shape).to(self.device)
        return t.long()

    def check_shape(self, x):
        return x if len(x.shape) == (self.ndims + 1) else x.unsqueeze(0)

    def miu_selection(self, i):
        
        i = int(i / self.n_queries * 10000)
        if 10 < i <= 50:
            miu = self.miu_init / 2
        elif 50 < i <= 200:
            miu = self.miu_init / 4
        elif 200 < i <= 500:
            miu = self.miu_init / 8
        elif 500 < i <= 1000:
            miu = self.miu_init / 16
        elif 1000 < i <= 2000:
            miu = self.miu_init / 32
        elif 2000 < i <= 4000:
            miu = self.miu_init / 64
        elif 4000 < i <= 6000:
            miu = self.miu_init / 128
        elif 6000 < i <= 8000:
            miu = self.miu_init / 256
        elif 8000 < i:
            miu = self.miu_init / 512
        else:
            miu = self.miu_init

        return miu

    def sh_selection(self, it):
        """ schedule to decrease the parameter p """

        t = max((float(self.n_queries - it) / self.n_queries - .0) ** 1., 0) * .75

        return t

    def get_init_patch_content(self, c, s, n_iter=1000):
     
        patch_univ = torch.zeros([1, c, s, s]).to(self.device)
        for _ in range(n_iter):
            size_init = torch.randint(low=1, high=math.ceil(s ** .5), size=[1]).item()
            loc_init = torch.randint(s - size_init + 1, size=[2])
            patch_univ[0, :, loc_init[0]:loc_init[0] + size_init, loc_init[1]:loc_init[1] + size_init] = 0.
            patch_univ[0, :, loc_init[0]:loc_init[0] + size_init, loc_init[1]:loc_init[1] + size_init
            ] += self.random_choice([c, 1, 1]).clamp(0., 1.)
            # self.show_image(patch_univ[0])

        return patch_univ.clamp(0., 1.)

    def show_image(self, x0):
        tmmiu = x0.clone()
        tmmiu = tmp.transpose(0, 1)
        tmmiu = tmp.transpose(1, 2)
        tmmiu = tmp.cpu()
        plt.imshow(tmp)
        plt.show()


    def Cross_shaped_patch(self, c, s, counter, x_new, loc, patches_content):
        width = self.width
        length = self.length
        tmp = patches_content.reshape(c, -1)
        left = tmp[:, 0: width*length].reshape(c, -1)
        right = tmp[:, -1*width*length:].reshape(c, -1)
        for i in range(width):
            loc_0 = loc[counter, 0]+i
            loc_1 = loc[counter, 1]
            index_x = list(range(loc_0, loc_0 + length, 1))
            index_y = list(range(loc_1, loc_1 + length, 1))
            x_new[counter, :, index_x, index_y] = left[:,i*length: (i+1)*length]
        for i in range(width):
            loc_0 = loc[counter, 0] + i
            loc_1 = loc[counter, 1] + length
            index_x = list(range(loc_0, loc_0 + length, 1))
            index_y = list(range(loc_1 - length, loc_1, 1))[::-1]
            x_new[counter, :, index_x, index_y] = right[:,i*length: (i+1)*length]

   
        return x_new[counter]

    def calc_n_pert_pixel(self):
        width = self.width
        length = self.length
        ## calculate the number of patch pixls
        # even
        if length % 2 == 0:
            n_intersection = 0
            while(width > 1):
                n_intersection = n_intersection + 2 * (width - 1)
                width = width - 2
        # odd
        else:
            n_intersection = width
            while(width > 1):
                width = width - 2
                n_intersection = n_intersection + 2 * (width)
        n_pert_pix = self.width * self.length * 2 - n_intersection
        return n_pert_pix

    def attack(self, x, y):
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        with torch.no_grad():
            adv = x.clone()
            c, h, w = x.shape[1:]
            n_features = c * h * w
            n_ex_total = x.shape[0]
            
            
            width = self.width
            length = self.length        


            n_pert_pix = self.calc_n_pert_pixel()
            print(n_pert_pix) 
            
            s = int(math.ceil(n_pert_pix ** .5))

            x_best = x.clone()
            x_new = x.clone()
            
            # random gernerate the of cordinate of Cross shaped patch 
            loc = torch.randint(low=0, high=h - (length + width - 1) + 1, size=[x.shape[0], 2])
            self.logger.log('start initialization...')
            # initial patch content 
            patches_content= torch.zeros([x.shape[0], c, s, s]).to(self.device)
            for counter in range(x.shape[0]):
                patches_content[counter] += self.get_init_patch_content(c, s).squeeze().clamp(0., 1.)

            # put the content on the patch region to generate the new candidate perturbed image.
            for counter in range(x.shape[0]):
                x_new[counter] = self.Cross_shaped_patch(c, s, counter, x_new, loc, patches_content[counter])

            assert abs(self.interval) > 1
          

            margin_min, loss_min = self.margin_or_ce(x_new, y)
            n_queries = torch.ones(x.shape[0]).to(self.device)
            self.logger.log('start iterative update...')
            for i in range(1, self.n_queries):
                # check points still to fool
                idx_to_fool = (margin_min > -1e-6).nonzero().squeeze()
                x_curr = self.check_shape(x[idx_to_fool])
                pcontent_curr = self.check_shape(patches_content[idx_to_fool])
                y_curr = y[idx_to_fool]
                margin_min_curr = margin_min[idx_to_fool]
                loss_min_curr = loss_min[idx_to_fool]
                loc_curr = loc[idx_to_fool]
                if len(y_curr.shape) == 0:
                    y_curr.unsqueeze_(0)
                    margin_min_curr.unsqueeze_(0)
                    loss_min_curr.unsqueeze_(0)

                    loc_curr.unsqueeze_(0)
                    idx_to_fool.unsqueeze_(0)

                # sample update
                l_i = int(max(self.miu_selection(i) ** .5 * s, 1))
                stripes_loc = torch.randint(s - l_i + 1, size=[2])

                pcontent_new = pcontent_curr.clone()
                x_new = x_curr.clone()
                loc_new = loc_curr.clone()
                update_loc = int(i % self.interval == 0)
                update_patch = 1. - update_loc
                if self.interval < 0 :
                    update_loc = 1. - update_loc
                    update_patch = 1. - update_patch
                for counter in range(x_curr.shape[0]):
                    if update_patch == 1.:
                        # update content
                        if l_i > 1:
                            pcontent_new[counter, :, stripes_loc[0]:stripes_loc[0] + l_i,
                            stripes_loc[1]:stripes_loc[1] + l_i] += self.random_choice([c, 1, 1])
                        else:
                            # make sure to sample a different color
                            old_clr = pcontent_new[counter, :, stripes_loc[0]:stripes_loc[0] + l_i,
                                      stripes_loc[1]:stripes_loc[1] + l_i].clone()
                            new_clr = old_clr.clone()
                            while (new_clr == old_clr).all().item():
                                new_clr = self.random_choice([c, 1, 1]).clone().clamp(0., 1.)
                            pcontent_new[counter, :, stripes_loc[0]:stripes_loc[0] + l_i,
                            stripes_loc[1]:stripes_loc[1] + l_i] = new_clr.clone()
                      

                        pcontent_new[counter].clamp_(0., 1.)
                    if update_loc == 1:
                        # update location
                        loc_new[counter] = torch.randint(low=0, high=h - (length + width - 1) + 1,  size=[2])


                    x_new[counter] = self.Cross_shaped_patch(c, s, counter, x_new, loc_new, pcontent_new[counter])
                 

                # check loss of new candidate
                margin, loss = self.margin_or_ce(x_new, y_curr)
                # print(margin[0])
                n_queries[idx_to_fool] += 1

                # update best solution
                idx_improved = (loss < loss_min_curr).float()
                idx_to_update = (idx_improved > 0.).nonzero().squeeze()
                loss_min[idx_to_fool[idx_to_update]] = loss[idx_to_update]

                idx_miscl = (margin < -1e-6).float()
                idx_improved = torch.max(idx_improved, idx_miscl)
                nimpr = idx_improved.sum().item()
                if nimpr > 0.:
                    idx_improved = (idx_improved.view(-1) > 0).nonzero().squeeze()
                    margin_min[idx_to_fool[idx_improved]] = margin[idx_improved].clone()
                    patches_content[idx_to_fool[idx_improved]] = pcontent_new[idx_improved].clone()
                    loc[idx_to_fool[idx_improved]] = loc_new[idx_improved].clone()

                # log results current iteration
                ind_succ = (margin_min <= 0.).nonzero().squeeze()
                if ind_succ.numel() != 0:
                    self.logger.log(' '.join(['{}'.format(i + 1),
                                              '- success rate={}/{} ({:.2%})'.format(
                                                  ind_succ.numel(), n_ex_total,
                                                  float(ind_succ.numel()) / n_ex_total),
                                              '- avg # queries={:.2f}'.format(
                                                  n_queries[ind_succ].mean().item()),
                                              '- med # queries={:.2f}'.format(
                                                  n_queries[ind_succ].median().item()),
                                              '- loss={:.3f}'.format(loss_min.mean()),
                                              '- max pert={:.0f}'.format(((x_new - x_curr).abs() > 0
                                                                          ).max(1)[0].view(x_new.shape[0], -1).sum(
                                                  -1).max()),
                                              '{}'.format(' - loc' if update_loc == 1. else ''),
                                              '- l_i={:}'.format(l_i),
                                              '{}'.format(' - improves' if nimpr > 0. else ''),
                                              ]))

                if ind_succ.numel() == n_ex_total:
                    break

            # apply patches
            for counter in range(x.shape[0]):
                x_best[counter] = self.Cross_shaped_patch(c, s, counter, x_best, loc, patches_content[counter])
        return n_queries, x_best
