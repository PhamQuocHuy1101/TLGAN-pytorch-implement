import os
import torch
from tqdm.auto import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config as cf
from model import Generator, Discriminator
from data.dataset import WrapData
from data.transform import to_tensor
import utils


# model
gen = Generator(cf.img_dim, cf.map_img_dim, cf.gen_hidden_dim)
gen.to(device = cf.device)
disc = Discriminator(cf.map_img_dim, cf.img_size)
disc.to(device = cf.device)

opt_gen = optim.AdamW(params = gen.parameters(), lr = cf.lr, betas=(cf.beta_1, cf.beta_2))
opt_disc = optim.AdamW(params = disc.parameters(), lr = cf.lr, betas=(cf.beta_1, cf.beta_2))

last_checkpoint = os.path.join(cf.checkpoint_path, f'{cf.last_step}_state_dict.pt')
if cf.continue_training and os.path.exists(last_checkpoint):
    checkpoint = torch.load_state_dict(last_checkpoint, map_location = cf.device)
    gen.load_state_dict(checkpoint['generator'])
    disc.load_state_dict(checkpoint['discriminator'])
    opt_gen.load_state_dict(checkpoint['opt_gen'])
    opt_disc.load_state_dict(checkpoint['opt_disc'])
    last_step = checkpoint['step']
    last_epoch = checkpoint['epoch']
else:
    gen.apply(utils.init_weight)
    disc.apply(utils.init_weight)
    last_step = 0
    last_epoch = -1

scheduler_g = optim.lr_scheduler.ExponentialLR(opt_gen, gamma=cf.lr_decay, last_epoch=last_epoch)
scheduler_d = optim.lr_scheduler.ExponentialLR(opt_disc, gamma=cf.lr_decay, last_epoch=last_epoch)

# data
files = sorted(os.listdir(os.path.join(cf.image_root)))
dataset = WrapData(files, cf.image_root, cf.map_root, to_tensor)
train_loader = DataLoader(dataset, batch_size=cf.batch_size, shuffle=True)

# logs
os.makedirs(cf.log_files, exist_ok=True)
sw = SummaryWriter(cf.log_files)

step = 0
gen.train()
disc.train()
for epoch in range(last_epoch + 1, cf.n_epochs, 1):
    print(f'Epoch {epoch} ==================================================')

    for r_img, r_map_img in tqdm(train_loader):
        #print(r_img.shape, r_map_img.shape)
        r_img = r_img.to(device = cf.device)
        r_map_img = r_map_img.to(device = cf.device)
        cur_batch_size = len(r_img)

        # disc
        disc_loss_item = 0.0
        f_map_img = gen(r_img)
        
        for _ in range(cf.crit_repeats):
            opt_disc.zero_grad()

            disc_r_map_img = disc(r_map_img)
            disc_f_map_img = disc(f_map_img.detach())

            # Gradient penalty
            # eps = torch.randn(cur_batch_size, 1, 1, 1).to(device = cf.device)
            # grad = utils.compute_gradient(disc, r_map_img, f_map_img, eps)
            # grad_loss = cf.grad_penalty * utils.gradient_penalty(grad)
            # loss = utils.disc_loss_gp(disc_r_map_img, disc_f_map_img, grad_loss)

            loss = utils.disc_loss(disc_r_map_img, disc_f_map_img)

            loss.backward(retain_graph = True)
            opt_disc.step()
            disc_loss_item += loss.item()
        disc_loss_item /= cf.crit_repeats
        
        # gen
        opt_gen.zero_grad()
        f_map_img = gen(r_img)
        disc_f_map_img = disc(f_map_img)
        gen_loss_item = utils.gen_loss(disc_f_map_img)
        gen_loss_item.backward()
        opt_gen.step()

        sw.add_scalar('loss/disciminator', disc_loss_item, step)
        sw.add_scalar('loss/generator', gen_loss_item.item(), step)

        step += 1
        if step % cf.checkpoint_interval == 0:
            print('Discriminator loss: ', disc_loss_item)
            print('Generator loss: ', gen_loss_item.item())
            
            # save checkpoint
            file_name = os.path.join(cf.checkpoint_path, f'{step}_state_dict.pt')
            torch.save({
                'generator': gen.state_dict(),
                'discriminator': disc.state_dict(),
                'opt_gen': opt_gen.state_dict(),
                'opt_disc': opt_disc.state_dict(),
                'step': step,
                'epoch': epoch
            }, file_name)
    
    scheduler_d.step()
    scheduler_g.step()
