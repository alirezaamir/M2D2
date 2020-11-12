DEBUG = True
SEG_N = 1024
if DEBUG:
    latent_dim_list = [16]
    beta_list       = [1e-3]
    lr_list         = [1e-8]
    decay_list      = [1e-5]
else:
    latent_dim_list = [1, 2, 4, 16]
    beta_list       = [1e-7, 1e-5, 1e-3, 1e-2]
    lr_list         = [1e-7, 1e-5]
    decay_list      = [1e-5, 1e-3]
