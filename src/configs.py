from numpy import False_

config_FIAE_500239 = {
    # "database": ['CHEDAR', 'HUTUBS', 'CIPIC'],
    "database": ["HUTUBS"],
    "sub_index": {
        "CIPIC": {
            "train": range(0, 33),  # 32
            "valid": range(33, 38),  # 6
            "test": range(38, 45),  # 7
            "all": range(0, 45),  # 45 subjects
        },
        "HUTUBS": {
            "train": range(0, 77),  # 77
            "valid": range(77, 87),  # 10
            "test": range(88, 95),  # 7, remove duplication: 95==0, 87==21
            "all": range(0, 96),  # 94 subjects
        },
        # 'CHEDAR': {
        #     'train': range(   0, 1003), # 1003
        #     'valid': range(1003, 1128), # 125
        #     'test':  range(1128, 1253), # 125
        #     'all':   range(   0, 1253), # 1253 subjects
        # },
    },
    "idx_plot_list": {  # horizontal plane f, l, b, r
        # 'CHEDAR': [0, 1, 2, 3],
        "HUTUBS": [202, 211, 220, 229],
        "CIPIC": [0, 1, 2, 3],
    },
    "activation_function": "nn.Mish()",
    "data_kind_interp": ["HRTF_mag"],
    "data_kind_hyper_en": ["SrcPos_Cart", "Freq", "B_mes"],
    "data_kind_hyper_de": ["SrcPos_Cart", "Freq"],
    "dim_data_hyper": {"SrcPos_Cart": 3, "Freq": 1, "B_mes": 1},
    "mid_mean_dim": (1,),
    # 'DNN_for_interp_ITD': False,
    # 'DNN_for_interp_HRIR': False,
    "HRIR_std_unit": False,
    "use_bias_FIAE_en": False,
    "use_bias_FIAE_de": False,
    "pinv_FIAE_en": True,
    "en_1_linear": True,
    "aggregation_mean": True,
    "weight_en_sph_harm": False,
    "weight_de_sph_harm": False,
    # "use_freq_for_hyper_en": True,
    # "use_Bp_for_hyper_en": True, #
    # "use_freq_for_hyper_de": True,
    "use_RSH_for_hyper_en": False,
    "use_RSH_for_hyper_de": False,
    # --- FourierFeatureMapping: FFM -----------
    "data_kind_ffm": ["SrcPos_Cart", "Freq"],  # B_mes
    "num_ff": {"SrcPos_Cart": 16, "Freq": 8, "B_mes": 16},
    # "use_ffm_for_hyper_cartpos": True,
    # "num_ff_cartpos": 32,
    "SrcPos_Cart_norm": 1.5,
    # - - -
    # "use_ffm_for_hyper_freq": True,
    # "num_ff_freq": 16,
    # - - -
    # "use_ffm_for_hyper_Bp": False,
    # "num_ff_Bp": 16,
    "Bp_norm": 865,
    # - - -
    "ffm_trainable": True,
    # -------------
    "hyper_identity_en": False,
    "hyper_identity_de": False,
    "de_2_skip": True,
    "max_truncation_order": 7,
    "dim_z": 64,
    "dim_z_af": 64,
    # - - - - - - - -
    "several_num_pts": True,
    "num_pts_list": list([2522]),  # 2522
    "num_pts": 2522,  # 2522
    "random_sample": True,
    "pln_smp": False,
    "pln_smp_axes": [0],
    "pln_smp_paral": False,
    "pln_smp_paral_axis": 2,
    "pln_smp_paral_values": [-0.735, 0, 0.735],
    "use_own_hrtf_test": False,
    "own_hrtf_sub_name": [
        "ito_yuki",
        "shigemi_kazuhide",
        "koyama_shoichi",
        "ueno_natsuki",
        "iijima_naoto",
    ]
    + ["imamura_kanami", "kitamura_makito", "seki_kentaro", "yamano_kota"]
    + [
        "arai_kohei",
        "kitamura_kota",
        "kuribayashi_masaki",
        "matsuoka_hiroyasu",
        "okita_ayumu",
        "takemoto_wataru",
        "watanabe_yuki",
    ],  # ['ito', 'shigemi', 'koyama', 'ueno', 'iijima'],
    "use_mit_kemar_test": False,
    # - - - -
    "reg_mat_base": "duraiswami",  # duraiswami, identity
    "cstr_relax": True,
    "reg_mat_learn": "None",  # "diag", # diag, full, None
    "reg_w": 1e-3,
    "rel_l": 1e2,
    "learning_rate": 1e-3,  # 0.01
    "epochs": 1400,
    "lr_update": "step",
    "lr_milestones": [800, 1200],
    "lr_gamma": 0.1,
    "batch_size": 1,  # 4,
    # ---------------
    # =========================
    "use_hypernet": True,
    "channel_En_0": 16,
    "channel_En_z": 128,
    "channel_De_z": 128,
    "channel_De_-1": 16,
    "channel_hyper": 64,
    "En_use_res": False,
    "De_use_res": False,
    "hyper_use_res": True,
    "hlayers_En_0": 2,
    "hlayers_En_z": 0,
    "hlayers_De_z": 0,
    "hlayers_De_-1": 2,
    "hlayers_hyper": 2,
    "Decoder": "hrtf_hyper",
    "coeff_skipconnection": False,
    "use_lr_aug": True,
    # =============
    "pos_all_en_attn": False,
    "pos_all_en_attn_gram": False,
    "pos_all_en_attn_eye_coeff": 1e-1,
    "channel_En_attn_k": 64,
    "channel_En_attn_q": 64,
    "hlayers_En_attn_k": 3,
    "hlayers_En_attn_q": 3,
    # =============
    "hyperlinear_en_0": True,
    "hyperconv_en_0": False,
    "hyperconv_FD_en_0": False,
    "hyperconv_en_0_ks": 3,
    "hyperconv_en_0_pad": "same",
    "hyperlinear_de_-1": True,
    "hyperconv_de_-1": False,
    "hyperconv_FD_de_-1": False,
    "hyperconv_de_-1_ks": 3,
    "hyperconv_de_-1_pad": "same",
    # =============
    "freeze_de": False,
    "green": False,  # nogreen
    "RIEC": False,
    "alpha": 0.2,
    "balanced": False,
    # =============
    "Encoder_add_fc": True,
    "z_norm": False,
    # =============
    "windowfunc": False,
    "window": "square",
    "out_nowindow": True,
    # =========================
    "droprate": 0.0,
    "fft_length": 256,
    # =========================
    "in_coeff": False,
    "in_magphase": False,
    "out_magphase": False,
    "in_mag": True,  #
    "out_mag": True,  #
    # --- PCA ---
    "in_mag_pc": False,
    "out_mag_pc": False,
    "num_pc": 128,
    # --- in/out latent --- PCAのかわり---
    "in_latent": False,
    "out_latent": False,
    "num_latents": 128,
    "hlayers_latents": 3,
    "channel_latents": 64,
    # --- in/out cnn --- PCAのかわり---
    "in_cnn": False,
    "out_cnn": False,
    "hlayers_cnn_pb": 1,  # layers/block (1block,1pooling)
    "blocks_cnn": 3,
    "ks_cnn": 3,
    "ks_cnn_plg": 2,
    "pooling_cnn": "stcv",  # ["max", "avg", "stcv"]
    "stride_cnn": 2,
    "channel_cnn": 16,
    # --- in/out cnn_ch --- ch 方向に補間 --- 他パラメタは in/out cnn の所で設定
    "in_cnn_ch": False,
    "out_cnn_ch": False,
    "use_freq_for_cnn_ch": False,
    "LN_dim": [-2, -1],  # [-2,-1] or [-1]
    "channel_inout": 1,  # default:1
    # ---
    "use_cae": False,
    "kernel_size": [3, 3],
    "num_layers_res": 10,
    "z_flatten": False,
    "use_unet": False,
    # =========================
    "lininv_experiment": False,
    "lininv_only": False,
    "loss_weights": {
        "mae_itd": 2.5e3,
        "mae_ild": 0,
        "lsd": 1,
    },  #
    "npwlsd_gamma": 0,
    "npwlsd_epsilon": 2,
    "is_pm_bottom": "target",
    "mask_beginning": 0,
    "max_frequency": 20000,
    # ============
    "fs_upsampling": 8 * 48000,
    "minphase_recon": True,
    "use_itd_gt": False,
    # ============
    "np_diff_loss": False,
    "window_length_np_diff": 64,
    "ls_diff_gen_order": 2,  # 1 or 2 or 4
    "ls_diff_gen_activation": "leakysign",  # [None,"sign","tanh","leakysign","leakytanh"]
    "ls_diff_gen_activate_coeff": 1.0,
    "ls_diff_gen_leak_coeff": 0.1,
    # ============
    "use_mid_conv": False,
    "use_mid_conv_simple": False_,
    "use_freq_for_mid_conv": False,
    "hlayers_mid_conv": 3,
    "channel_mid_conv": 64,
    "ch_mid_conv_simple": 64,
    "ks_mid_conv": 5,
    "res_mid_conv": True,
    # ============
    "use_mid_linear": False,
    # ============
    "use_metric": False,
    "metric_before": False,
    "num_cluster": 64,
    "class": "cluster",
    "use_attention": False,
    "pos2weight": True,
    "hlayers_p2w": 3,
    # ============
    "num_classifier": 1,
    "metric_margin": 0.25,  # 0.5
    "metric_scale": 30,
    # ============
    "model": "FIAE",
    "newbob_decay": 0.5,
    "newbob_max_decay": 1e-06,
    "normalize": "ln",
    "num_gpus": 2,
    "num_sub": 77,
    "only_left": False,
    "save_frequency": 500,
    "std_dim": [1, 3],
    "timestamp": "",
    "underdet": False,
    "use_freq_as_input": False,
    "use_nf": False,
}
