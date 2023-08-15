import os
from yacs.config import CfgNode
import argparse


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gpu', type=int, help='gpu id')
    parser.add_argument('-l', '--last_epoch', type=int, help='last epoch')
    parser.add_argument('-c', '--save_checkpoint', action='store_true', help='save temporary files')
    parser.add_argument('-m', '--model_name', type=str, help='model name')
    parser.add_argument('--config_path', type=str, default='./config', help='config path')
    parser.add_argument('--config_file', type=str, default='train_graph.yaml', help='config filename')

    opts = parser.parse_args()
    opts_dict = vars(opts)
    opts_list = []
    for key, value in zip(opts_dict.keys(), opts_dict.values()):
        if value is not None:
            opts_list.append(key)
            opts_list.append(value)

    yaml_file = os.path.join(opts.config_path, opts.config_file)
    cfg = CfgNode.load_cfg(open(yaml_file))
    cfg.merge_from_list(opts_list)

    # cfg.model_name = f'{cfg.model_name}.pkl'
    cfg.logdir = f'{cfg.logdir}'
    cfg.checkpointdir = f'{cfg.checkpointdir}'
    cfg.heatmap_size = tuple(cfg.heatmap_size)
    cfg.freeze()

    for k, v in cfg.items():
        print(f'{k}: {v}')
    return cfg

