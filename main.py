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


def main(args):
    batch_size = 16

    results_dir = './results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    fig_dir = './fig'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # BitFusion configuration file
    config_file = 'bf_e_conf.ini'
    # Create simulator object
    verbose = args.debug
    bf_e_sim = Simulator(config_file, verbose)
    bf_e_energy_costs = bf_e_sim.get_energy_cost()
    print(bf_e_sim)

    energy_tuple = bf_e_energy_costs
    print('')
    print('*'*50)
    print(energy_tuple)

    sim_sweep_columns = ['N', 'M',
                         'Max Precision (bits)', 'Min Precision (bits)',
                         'Network', 'Layer',
                         'Cycles', 'Memory wait cycles',
                         'WBUF Read', 'WBUF Write',
                         'OBUF Read', 'OBUF Write',
                         'IBUF Read', 'IBUF Write',
                         'DRAM Read', 'DRAM Write',
                         'Bandwidth (bits/cycle)',
                         'WBUF Size (bits)', 'OBUF Size (bits)', 'IBUF Size (bits)',
                         'Batch size']

    bf_e_sim_sweep_csv = os.path.join(
        results_dir, 'bitfusion-eyeriss-sim-sweep.csv')
    if os.path.exists(bf_e_sim_sweep_csv):
        bf_e_sim_sweep_df = pandas.read_csv(bf_e_sim_sweep_csv)
    else:
        bf_e_sim_sweep_df = pandas.DataFrame(columns=sim_sweep_columns)
    print('Got BitFusion Eyeriss, Numbers')

    bf_e_results = check_pandas_or_run(bf_e_sim, bf_e_sim_sweep_df,
                                       bf_e_sim_sweep_csv, batch_size=batch_size)
    bf_e_results = bf_e_results.groupby('Network', as_index=False).agg(np.sum)
    import ipdb
    ipdb.set_trace()
    area_stats = bf_e_sim.get_area()
    print(area_stats)

    print('BitFusion-Eyeriss comparison')
    eyeriss_area = 3.5*3.5*45*45/65./65.
    print('Area budget = {}'.format(eyeriss_area))

    if abs(sum(area_stats)-eyeriss_area)/eyeriss_area > 0.1:
        print('Warning: BitFusion Area is outside 10% of eyeriss')
    print('total_area = {}, budget = {}'.format(sum(area_stats), eyeriss_area))
    bf_e_area = sum(area_stats)

    baseline_data = []

    for bench in benchmarks.benchlist:
        lookup_dict = {'Benchmark': bench}

        bf_e_stats = df_to_stats(
            bf_e_results.loc[bf_e_results['Network'] == bench])
        bf_e_cycles = bf_e_stats.total_cycles * (batch_size / 16.)
        bf_e_time = bf_e_cycles / 500.e3 / 16
        bf_e_energy = bf_e_stats.get_energy(
            bf_e_sim.get_energy_cost()) * (batch_size / 16.)
        bf_e_power = bf_e_energy / bf_e_time * 1.e-9

        print('*'*50)
        print('Benchmark: {}'.format(bench))
        # print('Eyeriss time: {} ms'.format(eyeriss_time))
        print('BitFusion time: {} ms'.format(bf_e_time))
        # print('Eyeriss power: {} mWatt'.format(eyeriss_power*1.e3*16))
        print('BitFusion power: {} mWatt'.format(bf_e_power*1.e3*16))
        print('*'*50)

    eyeriss_comparison_df = pandas.DataFrame(
        baseline_data, columns=['Metric', 'Network', 'Value'])


def get_eyeriss_energy(df):
    eyeriss_energy_per_mac = 16 * 0.2 * 1.e-3  # energy in nJ
    eyeriss_energy_alu = float(df['ALU'])
    eyeriss_energy_dram = float(df['DRAM']) * 0.15  # Scaling due to technology
    eyeriss_energy_buffer = float(df['Buffer'])
    eyeriss_energy_array = float(df['Array'])
    eyeriss_energy_rf = float(df['RF'])
    eyeriss_energy = eyeriss_energy_alu + eyeriss_energy_dram + \
        eyeriss_energy_buffer + eyeriss_energy_array + eyeriss_energy_rf
    eyeriss_energy *= eyeriss_energy_per_mac
    return eyeriss_energy


def get_eyeriss_energy_breakdown(df):
    eyeriss_energy_per_mac = 16 * 0.2 * 1.e-3  # energy in nJ
    eyeriss_energy_alu = float(df['ALU'])
    eyeriss_energy_dram = float(df['DRAM'])
    eyeriss_energy_buffer = float(df['Buffer'])
    eyeriss_energy_array = float(df['Array'])
    eyeriss_energy_rf = float(df['RF'])
    eyeriss_energy = [eyeriss_energy_alu+eyeriss_energy_array,
                      eyeriss_energy_buffer, eyeriss_energy_rf, eyeriss_energy_dram]
    eyeriss_energy = [x * eyeriss_energy_per_mac for x in eyeriss_energy]
    return eyeriss_energy


def df_to_stats(df):
    stats = Stats()
    stats.total_cycles = float(df['Cycles'])
    stats.mem_stall_cycles = float(df['Memory wait cycles'])
    stats.reads['act'] = float(df['IBUF Read'])
    stats.reads['out'] = float(df['OBUF Read'])
    stats.reads['wgt'] = float(df['WBUF Read'])
    stats.reads['dram'] = float(df['DRAM Read'])
    stats.writes['act'] = float(df['IBUF Write'])
    stats.writes['out'] = float(df['OBUF Write'])
    stats.writes['wgt'] = float(df['WBUF Write'])
    stats.writes['dram'] = float(df['DRAM Write'])
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BitFusion Simulating')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='enable debug mode')
    args = parser.parse_args()
    main(args)
