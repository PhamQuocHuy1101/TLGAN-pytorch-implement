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


for epoch in range(cf.n_epochs):
    print(f'Epoch {epoch} ==================================================')

    for image, map_image in train_loader:
        for _ in range(cf.crit_repeats):
            pass