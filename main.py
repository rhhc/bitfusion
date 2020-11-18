import argparse
import pandas
import os
import numpy as np
import sys
import csv
import dnnweaver2
import bitfusion.src.benchmarks.benchmarks as benchmarks
from bitfusion.src.simulator.stats import Stats
from bitfusion.src.simulator.simulator import Simulator
from bitfusion.src.sweep.sweep import SimulatorSweep, check_pandas_or_run
from bitfusion.src.utils.utils import *
from bitfusion.src.optimizer.optimizer import optimize_for_order, get_stats_fast

import ipdb

def main(args):
    results_dir = './results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    assert os.path.exists(args.models), '{} does not exist'.format(args.models)

    # BitFusion configuration file
    config_file = 'bf_e_conf.ini'
    # Create simulator object
    bf_e_sim = Simulator(config_file, verbose=args.debug)
    bf_e_energy_costs = bf_e_sim.get_energy_cost()
    print(bf_e_sim)
    print(bf_e_energy_costs)
    stats, _ = bf_e_sim.get_conv_cycles(
        K=3, O=32, S=1, IC=32, OC=64, iprec=4, wprec=4, batch_size=1)
    stats, _ = bf_e_sim.get_conv_cycles(
        K=3, O=32, S=1, IC=32, OC=64, iprec=8, wprec=8, batch_size=1)
    stats, _ = bf_e_sim.get_FC_cycles(100, 10, 3, 3, batch_size=1)

    with open(os.path.join(results_dir, 'fc_layer_latency.csv'), 'w') as wf:
        info_deduplicate = set()
        for f_name in os.listdir(args.models):
            if '_fc' in f_name:
                print(f_name)
                df = pandas.read_csv(os.path.join(args.models, f_name))
                for index, row in df.iterrows():
                    info = (row['IC'], row['OC'])
                    info_deduplicate.add(info)
                    print('Add {}'.format(info))
        print(info_deduplicate)
        for i in info_deduplicate:
            print('processing {}'.format(i))
            for iprec in range(2,9):
                for wprec in range(2, 9):
                    stats, _  = bf_e_sim.get_FC_cycles(i[0], i[1], iprec=iprec, wprec=wprec)
                    print(stats.total_cycles)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BitFusion Simulating')
    parser.add_argument('--models', default='models', type=str,
                        help='the root directory to layer configuration files')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='enable debug mode')
    args = parser.parse_args()
    main(args)
