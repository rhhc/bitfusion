import pandas
import os
import numpy as np
import sys
import csv
sys.path.insert(0, '../dnnweaver2')
print(sys.version_info)
import dnnweaver2

import bitfusion.src.benchmarks.benchmarks as benchmarks
from bitfusion.src.simulator.stats import Stats
from bitfusion.src.simulator.simulator import Simulator
from bitfusion.src.sweep.sweep import SimulatorSweep, check_pandas_or_run
from bitfusion.src.utils.utils import *
from bitfusion.src.optimizer.optimizer import optimize_for_order, get_stats_fast

def main(index=0):
    batch_size = 16
    
    results_dir = './results-mobilenet'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    #  if last result exists, the simulator would no generate new results
    if index is None:
        index = 'whole'
    result_file = "bitfusion-abs-sim-sweep-%s.csv" % str(index)
    if os.path.exists(os.path.join(results_dir, result_file)):
        os.remove(os.path.join(results_dir, result_file))
    
    # BitFusion configuration file
    config_file = 'bf_e_conf.ini'
    
    # Create simulator object
    verbose = False
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
    
    bf_e_sim_sweep_csv = os.path.join(results_dir, result_file)
    if os.path.exists(bf_e_sim_sweep_csv):
        bf_e_sim_sweep_df = pandas.read_csv(bf_e_sim_sweep_csv)
    else:
        bf_e_sim_sweep_df = pandas.DataFrame(columns=sim_sweep_columns)
    print('Got BitFusion Eyeriss, Numbers')
    
    bf_e_results = check_pandas_or_run(bf_e_sim, bf_e_sim_sweep_df, bf_e_sim_sweep_csv, batch_size=batch_size)
    bf_e_results = bf_e_results.groupby('Network', as_index=False).agg(np.sum)
    area_stats = bf_e_sim.get_area()
    
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
    
    header = ["config", "latency(ms)", "power(mWatt)"]
    profile_result = "%s/layer-wise-%s.csv" % (results_dir, str(index))
    with open(profile_result, 'wb') as f:
        w = csv.writer(f)
        w.writerows([header])
    
        print("benchlist length %d" % len(benchmarks.benchlist))
        for bench in benchmarks.benchlist:
            if 'base' in bench:
                continue
    
            bf_e_stats = df_to_stats(bf_e_results.loc[bf_e_results['Network'] == bench])
            bf_e_cycles = bf_e_stats.total_cycles * (batch_size / 16.)
            bf_e_time = bf_e_cycles / 500.e3 / 16
            bf_e_energy = bf_e_stats.get_energy(bf_e_sim.get_energy_cost()) * (batch_size / 16.)
            #bf_e_power = bf_e_energy / bf_e_time * 1.e-9
    
            if 'layer' in bench:
                base_bf_e_stats = df_to_stats(bf_e_results.loc[bf_e_results['Network'] == bench.replace('layer', 'base')])
                base_bf_e_cycles = base_bf_e_stats.total_cycles * (batch_size / 16.)
                base_bf_e_time = base_bf_e_cycles / 500.e3 / 16
                base_bf_e_energy = base_bf_e_stats.get_energy(bf_e_sim.get_energy_cost()) * (batch_size / 16.)
                #base_bf_e_power = base_bf_e_energy / bf_e_time * 1.e-9
            else:
                base_bf_e_time = 0
                base_bf_e_energy = 0
    
            latency = bf_e_time - base_bf_e_time
            energy = bf_e_energy - base_bf_e_energy
            power = energy / latency * 1.e-9
    
            w.writerows([[bench, latency, power]])
    
            print('*'*50)
            print('Benchmark: {}'.format(bench))
            print('BitFusion time: {} ms'.format(latency))    
            print('BitFusion power: {} mWatt'.format(power*1.e3*16))
            print('*'*50)
    

if __name__ == '__main__':
    index = os.getenv('bitfusion_index')
    try:
        index = int(index)
    except:
        index = None
    index = -1
    print("Index", index)
    main(index)

