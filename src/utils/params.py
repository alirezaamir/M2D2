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


pat_list = ['pat_102', 'pat_16202', 'pat_11002', 'pat_8902', 'pat_7302', 'pat_26102', 'pat_23902', 'pat_22602',
            'pat_21902', 'pat_21602', 'pat_30002', 'pat_30802', 'pat_32502', 'pat_32702', 'pat_45402', 'pat_46702',
            'pat_55202', 'pat_56402', 'pat_58602', 'pat_59102', 'pat_103002', 'pat_92102', 'pat_93902', 'pat_85202',
            'pat_111902', 'pat_75202', 'pat_96002', 'pat_79502', 'pat_109502', 'pat_114902']


pat_hours = {'pat_102': 173,
             'pat_7302': 281,
             'pat_8902': 174}