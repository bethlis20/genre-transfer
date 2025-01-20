import argparse
import json
import os
from data_loader import get_dataloader
from solver import Solver

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', action='store', type=str, required=True, help='Path to config file')
    parser.add_argument('--num_epoch', action='store', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--resume', action='store', type=str, default='', help='Path to checkpoint file to resume from')
    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        print(f"Error: Config file {args.config_file} not found.")
        exit(1)

    config = json.load(open(args.config_file))

    train_loader = {}
    test_loader = {}
    train_loader['a'], train_loader['b'] = get_dataloader(a=config['class_a'], b=config['class_b'], training=True)
    test_loader['a'], test_loader['b'] = get_dataloader(a=config['class_a'], b=config['class_b'], training=False)

    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')

    solver = Solver(train_loader, test_loader, config)
    solver.train(args.num_epoch)
