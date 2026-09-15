import argparse
import os
import torch
from experiments.core.exp_forecasting import Exp_Long_Term_Forecast1
import random
import numpy as np
from pathlib import Path


MOE_MODELS = {'iMOE', 'iMOE_CSR', 'iMOE_HD', 'iMOE_SDR'}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_parser():
    parser = argparse.ArgumentParser(description='iMOE')

    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--is_training', type=int,  default=1, help='status')
    parser.add_argument('--model', type=str,  default='iMOE',
                        help='iMOE,iMOE_CSR,iMOE_HD,iMOE_SDR,ConditionedLSTM,ConditionedMLP,DegradationFormer,Informer,PATCHTST')

    parser.add_argument('--dataset', type=str, default='UL-NCA', help='')
    parser.add_argument('--condition', type=str, default='CY25-05_1', 
                        help='UL-NCA:CY45-05_1,CY25-05_1,CY25-025_1,CY25-1_1,CY35-05_1'
                        'UL-NCM:CY45-05_1,CY25-05_1,CY35-05_1,UL-NCMNCA:CY25-05_1,CY25-05_2,CY25-05_4,TPSL:Arbitrary,Fixed,LSD:LSD')
    parser.add_argument('--test_condition', type=str, default=None,
                        help='optional held-out condition for cross-condition evaluation')
    parser.add_argument('--seq_len', type=int, default=50, help='')
    parser.add_argument('--enc_in', type=int, default=1, help='input sequence length')
    parser.add_argument('--hidden_dim', type=int, default=64, help='')
    parser.add_argument('--pred_len', type=int, default=50, help='prediction horizon')
    parser.add_argument('--num_experts', type=int, default=5)
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--baseline_top_k', type=int, default=2)
    parser.add_argument('--csr_top_k', type=int, default=4)
    parser.add_argument('--curve_channels', type=int, default=4)
    parser.add_argument('--fusion_mode', choices=('learned', 'fixed'), default='learned',
                        help='iMOE_SDR router fusion: learned gate or fixed 0.5 average')
    parser.add_argument('--fusion_gate_bias', type=float, default=0.0,
                        help='initial bias of the learned iMOE_SDR fusion gate')
    parser.add_argument('--disable_noisy_routing', action='store_true',
                        help='ablate training-time router noise in iMOE_SDR')
    parser.add_argument('--disable_top_k', action='store_true',
                        help='ablate sparse top-k routing in iMOE_SDR')
    parser.add_argument('--alpha', type=int, default=10)
    parser.add_argument('--diverloss', type=float, default=0.5)
    parser.add_argument('--soc', type=int, default=20, help='20,30,40')
    parser.add_argument('--dataaccess', type=int, default=100,
                        help='legacy percentage selector; formal data-efficiency runs use --train_battery_count')
    parser.add_argument('--train_battery_count', type=int, default=None,
                        help='explicit number of training batteries for deterministic nested subsets')
    parser.add_argument('--data_root', type=Path,
                        default=PROJECT_ROOT / 'datasets' / 'processed',
                        help='processed dataset root')

    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.2, help='dimension of fcn')
    parser.add_argument('--patch_size', type=int, default=2, help='dimension of fcn')
    parser.add_argument('--disable_trend', action='store_true',
                        help='ablate the explicit trend component')
    parser.add_argument('--disable_residual', action='store_true',
                        help='ablate the learned residual component')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=1500,
                        help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=64,
                        help='test/inference batch size; does not change training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
    parser.add_argument('--checkpoints', type=str,
                        default=str(PROJECT_ROOT / 'checkpoints' / 'main'),
                        help='location of model checkpoints')
    parser.add_argument('--results_dir', type=str,
                        default=str(PROJECT_ROOT / 'results' / 'runs' / 'current'),
                        help='location of predictions, metrics, and plots')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='checkpoint file to load when --is_training=0')
    parser.add_argument('--patience', type=int, default=200, help='early-stopping patience')
    parser.add_argument('--inverse', type=str, default='no', help='s')
    return parser


def build_setting(args, iteration):
    if args.model in MOE_MODELS:
        if args.model == 'iMOE_SDR':
            setting = '{}_ds{}_ex{}_pl{}_btk{}_ctk{}_cc{}_dm{}'.format(
                args.model,
                args.dataset,
                args.num_experts,
                args.pred_len,
                args.baseline_top_k,
                args.csr_top_k,
                getattr(args, 'curve_channels', 4),
                iteration,
            )
        else:
            setting = '{}_ds{}_ex{}_pl{}_tk{}_dm{}'.format(
                args.model,
                args.dataset,
                args.num_experts,
                args.pred_len,
                args.top_k,
                iteration,
            )
    else:
        setting = '{}_ds{}_pl{}_dm{}'.format(
            args.model,
            args.dataset,
            args.pred_len,
            iteration,
        )

    setting += '_seed{}_da{}'.format(args.seed, args.dataaccess)
    if getattr(args, 'train_battery_count', None) is not None:
        setting += '_nb{}'.format(args.train_battery_count)
    setting += '_ep{}_bs{}'.format(
        getattr(args, 'train_epochs', 1500),
        getattr(args, 'batch_size', 32),
    )
    if getattr(args, 'des', 'test') != 'test':
        setting += '_des{}'.format(args.des)
    if getattr(args, 'test_condition', None):
        setting += '_test{}'.format(args.test_condition)
    if args.model == 'DegradationFormer':
        setting += '_trend{}_residual{}'.format(
            'off' if getattr(args, 'disable_trend', False) else 'on',
            'off' if getattr(args, 'disable_residual', False) else 'on',
        )
    if args.model == 'iMOE_SDR':
        setting += '_fusion{}_fgb{}_noise{}_topk{}'.format(
            getattr(args, 'fusion_mode', 'learned'),
            getattr(args, 'fusion_gate_bias', 0.0),
            'off' if getattr(args, 'disable_noisy_routing', False) else 'on',
            'off' if getattr(args, 'disable_top_k', False) else 'on',
        )
    return setting


def main():
    parser = build_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')

    Exp = Exp_Long_Term_Forecast1

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args) 
            setting = build_setting(args, ii)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = build_setting(args, ii)

        exp = Exp(args)  
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
