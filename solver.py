import torch
from torch import nn
import numpy as np
from torch.nn.modules import loss
from tqdm import tqdm

from models.discriminator import Discriminator
from models.generator import Generator
from models.embedder import Embedder
import losses

def split(tensor):
    return torch.split(tensor, 128, dim=-1)


def combine(tensors):
    return torch.cat(tensors, dim=-1)


class Solver():

    def __init__(self, train_loader, test_loader, config):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_loader = train_loader
        self.test_loader = test_loader

        # epoch
        self.epoch = 1

        # networks
        self.gen = Generator().to(self.device)
        self.dis = Discriminator().to(self.device)
        self.embed = Embedder(latent_dim=128).to(self.device)

        self.losses = {
            'gen': [],
            'dis': [],
        }

        self.gen_freq = config['gen_freq']

        self.gen_lr = config['optimizers']['gen_lr']
        self.dis_lr = config['optimizers']['dis_lr']
        self.beta1 = config['optimizers']['beta1']
        self.beta2 = config['optimizers']['beta2']
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), self.gen_lr, [self.beta1, self.beta2])
        self.dis_opt = torch.optim.Adam(self.dis.parameters(), self.dis_lr, [self.beta1, self.beta2])

        self.hparam = config['hparam']

        self.epoch_save = config['epoch_save']

        if config['resume'] != '':
            checkpoint = torch.load(config['resume'])
            self.epoch = checkpoint['epoch'] + 1
            self.gen.module.load_state_dict(checkpoint['gen'])
            self.dis.module.load_state_dict(checkpoint['dis'])
            self.embed.module.load_state_dict(checkpoint['embed'])
            self.losses = checkpoint['losses']

    def reset_grad(self):
        self.dis_opt.zero_grad()
        self.gen_opt.zero_grad()

    def train_dis(self, a, b):

        a = a.to(self.device)
        b = b.to(self.device)

        a1, a2, a3 = split(a)

        ab1 = self.gen(a1)
        ab2 = self.gen(a2)
        ab3 = self.gen(a3)
        ab = combine([ab1, ab2, ab3])

        dis_ab = self.dis(ab)
        dis_b = self.dis(b)

        loss_adv_dis = losses.adv_loss(dis_b, real=True) + losses.adv_loss(dis_ab, real=False)

        loss_dis = loss_adv_dis
        self.reset_grad()
        loss_dis.backward(retain_graph=True)
        self.dis_opt.step()

        return loss_dis.item()

    def train_gen(self, a, b):

        a = a.to(self.device)
        b = b.to(self.device)

        a1, a2, a3 = split(a)

        ab1 = self.gen(a1)
        ab2 = self.gen(a2)
        ab3 = self.gen(a3)
        ab = combine([ab1, ab2, ab3])

        dis_ab = self.dis(ab)

        emb_a1 = self.embed(a1)
        emb_a2 = self.embed(a2)
        emb_ab1 = self.embed(ab1)
        emb_ab2 = self.embed(ab2)

        loss_adv_gen = torch.mean(-dis_ab)
        loss_embed = losses.embed_loss(emb_a1, emb_a2, emb_ab1, emb_ab2)
        loss_margin = losses.margin_loss(emb_a1, emb_a2)

        loss_gen = self.hparam['lambda_adv'] * loss_adv_gen + self.hparam['lambda_embed'] * loss_embed + self.hparam['lambda_margin'] * loss_margin
        self.reset_grad()
        loss_gen.backward(retain_graph=True)
        self.gen_opt.step()

        return loss_gen.item(), loss_adv_gen.item(), loss_embed.item(), loss_margin.item()

    def train(self, num_epoch=100):

        # loop epoch
        while self.epoch <= num_epoch:

            print('Epoch {}'.format(self.epoch))

            losses_gen = []
            losses_gen_adv = []
            losses_embed = []
            losses_margin = []
            losses_dis = []

            # loop batch
            for idx, (a, b) in tqdm(enumerate(zip(self.train_loader['a'], self.train_loader['b'])), total=len(self.train_loader['a'])):
                if a.shape[0] != b.shape[0]:
                    continue
                
                # train discriminator
                loss_dis = self.train_dis(a, b)

                # train generator
                loss_gen = None
                loss_gen_adv = None
                loss_embed = None
                loss_margin = None
                if idx % self.gen_freq == 0:
                    loss_gen, loss_gen_adv, loss_embed, loss_margin = self.train_gen(a, b)

                losses_gen.append(loss_gen)
                losses_dis.append(loss_dis)
                losses_gen_adv.append(loss_gen_adv)
                losses_embed.append(loss_embed)
                losses_margin.append(loss_margin)

            losses_gen = [x for x in losses_gen if x is not None]
            losses_dis = [x for x in losses_dis if x is not None]
            losses_gen_adv = [x for x in losses_gen_adv if x is not None]
            losses_embed = [x for x in losses_embed if x is not None]
            losses_margin = [x for x in losses_margin if x is not None]
            self.losses['gen'].append(np.mean(losses_gen))
            self.losses['dis'].append(np.mean(losses_dis))

            print('  gen loss:', np.mean(losses_gen))
            print('      adv loss:   ', np.mean(losses_gen_adv))
            print('      embed loss: ', np.mean(losses_embed))
            print('      margin loss:', np.mean(losses_margin))
            print('  dis loss:', np.mean(losses_dis))

            # save checkpoint
            if self.epoch % 10 == 0:  
                torch.save({
                    'epoch': self.epoch,
                    'gen': self.gen.state_dict(),
                    'dis': self.dis.state_dict(),
                    'embed': self.embed.state_dict(),
                    'losses': self.losses
                }, f'./checkpoints/checkpoint_{self.epoch}.pt')

            self.epoch += 1
