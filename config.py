# optim
device = 'cuda:0'
n_epochs = 100
z_dim = 64
display_step = 50
batch_size = 128
lr = 0.0002
beta_1 = 0.5
beta_2 = 0.999
c_lambda = 10
crit_repeats = 5
grad_penalty = 10
lr_decay = 10
checkpoint_interval = 100

# data
image_root = './data/croped_images/images'
map_root = './data/croped_images/map_images'

# checkpoint path
continue_training = False
last_step = -1
log_files = './checkpoint/logs'
checkpoint_path = './checkpoint'

# model
img_size = 128
img_dim  = 3
map_img_dim = 1
gen_hidden_dim = 16