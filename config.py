import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_config(case):
    if case == 'train_decoding': 
        parser = argparse.ArgumentParser(description='Specify config args for training EEG-To-Text decoder')
        
        parser.add_argument('-m', '--model_name', help='choose from {BrainTranslator, BrainTranslatorNaive, MultiViewConformerTranslator}', default = "BrainTranslator" ,required=True)
        parser.add_argument('-t', '--task_name', help='choose from {task1, task1_task2, task1_task2_task3, task1_task2_taskNRv2, task1_task2_task3_taskNRv2_taskTSRv2}', default = "task1", required=True)
        
        parser.add_argument('-1step', '--one_step', dest='skip_step_one', action='store_true')
        parser.add_argument('-2step', '--two_step', dest='skip_step_one', action='store_false')

        parser.add_argument('-pre', '--pretrained', dest='use_random_init', action='store_false')
        parser.add_argument('-rand', '--rand_init', dest='use_random_init', action='store_true')
        
        parser.add_argument('-load1', '--load_step1_checkpoint', dest='load_step1_checkpoint', action='store_true')
        parser.add_argument('-no-load1', '--not_load_step1_checkpoint', dest='load_step1_checkpoint', action='store_false')

        parser.add_argument('-ne1', '--num_epoch_step1', type = int, help='num_epoch_step1', default = 20, required=True)
        parser.add_argument('-ne2', '--num_epoch_step2', type = int, help='num_epoch_step2', default = 30, required=True)
        parser.add_argument('-lr1', '--learning_rate_step1', type = float, help='learning_rate_step1', default = 0.00005, required=True)
        parser.add_argument('-lr2', '--learning_rate_step2', type = float, help='learning_rate_step2', default = 0.0000005, required=True)
        parser.add_argument('-b', '--batch_size', type = int, help='batch_size', default = 32, required=True)

        parser.add_argument('-patience', '--patience', type=int, help='early stopping patience (epochs)', default=10)
        parser.add_argument('-no_early_stop', '--no_early_stop', action='store_true', help='disable early stopping')
        parser.add_argument('-lora_r', '--lora_r', type=int, help='LoRA rank', default=16)
        parser.add_argument('-lora_targets', '--lora_targets', nargs='+',
                            help='LoRA target modules',
                            default=['q_proj', 'k_proj', 'v_proj', 'out_proj'])
        parser.add_argument('-label_smooth', '--label_smooth', type=float,
                            help='label smoothing epsilon (0=off)', default=0.1)
        parser.add_argument('-lr_lora', '--lr_lora', type=float, default=None,
                            help='absolute LoRA learning rate in step 2; None keeps the '
                                 'historical LR2*2.0 coupling')
        parser.add_argument('-suffix', '--save_suffix', type=str, default='',
                            help='extra suffix appended to save_name; use it to keep runs that '
                                 'differ only in a parameter absent from save_name (e.g. lora_r '
                                 'or the architecture) from overwriting each other in '
                                 'config/decoding/')

        # ── MultiViewConformerTranslator architecture ────────────────────────
        # Every default reproduces the geometry that used to be hard-coded in
        # train_multiview.py and eval_multiview.py, so existing scripts build
        # exactly the same model they always did.
        parser.add_argument('-d_model', '--d_model', type=int, default=512,
                            help='width of the per-region and global transformers')
        parser.add_argument('-n_filters', '--n_filters', type=int, default=40,
                            help='temporal conv output channels')
        parser.add_argument('-n_spatial_filters', '--n_spatial_filters', type=int, default=None,
                            help='spatial conv output channels; None = same as n_filters '
                                 '(the historical behaviour)')
        parser.add_argument('-tk', '--temporal_kernel', type=int, default=200,
                            help='temporal conv kernel length')
        parser.add_argument('-pool', '--pool_stride', type=int, default=50,
                            help='AvgPool kernel over the raw time axis')
        parser.add_argument('-tpv', '--tokens_per_view', type=int, default=64,
                            help='local tokens per region after adaptive pooling')
        parser.add_argument('-cls', '--n_cls_per_view', type=int, default=8,
                            help='CLS tokens per region')
        parser.add_argument('-heads', '--n_heads', type=int, default=8,
                            help='attention heads, local and global')
        parser.add_argument('-enc_layers', '--n_encoder_layers', type=int, default=4,
                            help='per-region transformer depth')
        parser.add_argument('-glob_layers', '--n_global_layers', type=int, default=3,
                            help='global transformer depth')
        parser.add_argument('-dropout', '--dropout', type=float, default=0.1,
                            help='dropout inside the encoders and transformers')

        parser.add_argument('-s', '--save_path', help='checkpoint save path', default = './checkpoints/decoding', required=True)
        parser.add_argument('-subj', '--subjects', help='use all subjects or specify a particular one', default = 'ALL', required=False)
        parser.add_argument('-eeg', '--eeg_type', help='choose from {GD, FFD, TRT}', default = 'GD', required=False)
        parser.add_argument('-band', '--eeg_bands', nargs='+', help='specify freqency bands', default = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] , required=False)
        parser.add_argument('-cuda', '--cuda', help='specify cuda device name, e.g. cuda:0, cuda:1, etc', default = 'cuda:0')
        parser.add_argument('--resume', type=str, default=None,
                            help='path to merged checkpoint (.pt) to resume training from; skips encoder_best load and step1, runs step2 only with fresh LoRA')
        parser.add_argument('--no_lora', action='store_true',
                            help='disable LoRA in step2; unfreeze all BART params for full fine-tune (use with --resume to continue from a step1 or merged checkpoint)')
        
        args = vars(parser.parse_args())

    elif case == 'eval_decoding':
        parser = argparse.ArgumentParser(description='Specify config args for evaluate EEG-To-Text decoder')
        parser.add_argument('-checkpoint', '--checkpoint_path', help='specify model checkpoint' ,required=True)
        parser.add_argument('-conf', '--config_path', help='specify training config json' ,required=True)
        parser.add_argument('-cuda', '--cuda', help='specify cuda device name, e.g. cuda:0, cuda:1, etc', default = 'cuda:0')
        parser.add_argument('-tf', '--tf', help='use teacher forcing', default = True)
        parser.add_argument('-n', '--noise', help='use noise as input', default = True)
        args = vars(parser.parse_args())

    return args
