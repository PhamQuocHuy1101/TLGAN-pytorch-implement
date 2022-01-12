import os

import torch
from tqdm.auto import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

import config as cf
from model import Generator, Discriminator
from data.dataset import WrapData
from data.transform import to_tensor
import utils


# model
gen = Generator(3, 1, 16)
disc = Discriminator(1, 128)

gen.apply(utils.init_weight)
disc.apply(utils.init_weight)

opt_gen = optim.Adam(params = gen.parameters(), lr = cf.lr, betas=(cf.beta_1, cf.beta_2))
opt_disc = optim.Adam(params = disc.parameters(), lr = cf.lr, betas=(cf.beta_1, cf.beta_2))


# data
files = sorted(os.listdir(os.path.join(cf.image_root)))
dataset = WrapData(files, cf.image_root, cf.map_root, to_tensor)
train_loader = DataLoader(dataset, batch_size=cf.batch_size, shuffle=True)

gen_losses = []
disc_losses = []
for epoch in range(cf.n_epochs):
    print(f'Epoch {epoch} ==================================================')

    for r_img, r_map_img in tqdm(train_loader):
        r_img = r_img.to(device = cf.device)
        r_map_img = r_map_img.to(device = cf.device)
        cur_batch_size = len(r_img)

        # disc
        disc_loss_item = 0.0
        for _ in range(cf.crit_repeats):
            opt_disc.zero_grad()

            f_map_img = gen(r_map_img)
            disc_r_map_img = disc(r_map_img)
            disc_f_map_img = disc(f_map_img.detach())

            esp = torch.randn(cur_batch_size, 1, 1, 1).to(device = cf.device)
            grad = utils.compute_gradient(disc, r_map_img, f_map_img, esp)
            grad_loss = cf.grad_penalty * utils.gradient_penalty(grad)
            loss = utils.disc_loss(disc_r_map_img, disc_f_map_img, grad_loss)

            loss.backward(retain_graph = True)
            opt_disc.step()
            disc_loss_item += loss.item()
        disc_losses.append(disc_loss_item / cf.crit_repeats)