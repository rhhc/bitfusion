import argparse
import pandas as pd
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

    info_deduplicate_fc = set()
    info_deduplicate_conv = set()
    for f_name in os.listdir(args.models):
        if '_fc' in f_name:
            print(f_name)
            df = pd.read_csv(os.path.join(args.models, f_name))
            for index, row in df.iterrows():
                info = (row['IC'], row['OC'])
                info_deduplicate_fc.add(info)
                print('Add {}'.format(info))
        if '_conv' in f_name:
            print(f_name)
            df = pd.read_csv(os.path.join(args.models, f_name))
            for index, row in df.iterrows():
                info = (row['K'], row['O'], row['S'],
                        row['IC'], row['OC'])  # K,O,S,IC,OC
                info_deduplicate_conv.add(info)
                print('Add {}'.format(info))

    res_fc = pd.DataFrame(columns=('IC', 'OC', 'iprec', 'wprec', 'cycle'))
    res_conv = pd.DataFrame(
        columns=('K', 'O', 'S', 'IC', 'OC', 'iprec', 'wprec', 'cycle'))
    for i, info in enumerate(info_deduplicate_fc):
        print('simulating fc cycles {}/{} {}'.format(i +
                                                     1, len(info_deduplicate_fc), info))
        for iprec in range(2, 9):
            for wprec in range(2, 9):
                stats, _ = bf_e_sim.get_FC_cycles(
                    info[0], info[1], iprec=iprec, wprec=wprec)
                res_fc = res_fc.append(
                    [{'IC': info[0], 'OC':info[1], 'iprec':iprec, 'wprec':wprec, 'cycle':stats.total_cycles}], ignore_index=True)
                print(stats.total_cycles)
    res_fc.to_csv(os.path.join(results_dir, 'cycle_fc.csv'))
    for i, info in enumerate(info_deduplicate_conv):
        print('simulating conv cycles {}/{} {}'.format(i +
                                                       1, len(info_deduplicate_conv), info))
        for iprec in range(2, 9):
            for wprec in range(2, 9):
                stats, _ = bf_e_sim.get_conv_cycles(
                    K=info[0], O=info[1], S=info[2], IC=info[3], OC=info[4], iprec=iprec, wprec=wprec)
                res_conv = res_conv.append([{'K': info[0], 'O':info[1], 'S':info[2], 'IC':info[3], 'OC':info[4],
                                             'iprec':iprec, 'wprec':wprec, 'cycle': stats.total_cycles}], ignore_index=True)
    res_conv.to_csv(os.path.join(results_dir, 'cycle_conv.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BitFusion Simulating')
    parser.add_argument('--models', default='models', type=str,
                        help='the root directory to layer configuration files')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='enable debug mode')
    args = parser.parse_args()
    main(args)
