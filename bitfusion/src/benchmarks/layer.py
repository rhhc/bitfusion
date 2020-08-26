
import os, sys
import argparse
import logging
import re

from dnnweaver2.graph import Graph, get_default_graph
from dnnweaver2 import get_tensor
from dnnweaver2.scalar.dtypes import FQDtype, FixedPoint
from benchmarks import fc, conv

def obtainInt(l):
    lists = []
    for i in l:
        try:
            value = int(i)
        except ValueError:
            continue
        lists.append(value)
    return lists

def load_record(record="models/pytorch-resnet18.txt", result=set()):
    with open(record) as f:
        lines = f.readlines()
        f.close()
        total_count = 0
        for line in lines:
            if 'torch.Size' not in line:
                continue
            if line[0] == "#":
                continue
            items = re.split('\)|\(| |,|\[|\]', line)
            lists = obtainInt(items)
            assert len(lists) == 17, 'unexpected length'
            cin = lists[1]
            height = lists[2]
            width = lists[3]
            cout = lists[4]
            kernel = lists[8] 
            stride = lists[10]
            pad = lists[12]
            group = lists[16]

            count = 0
            for i in range(4, cin + 3 , 4):
                for j in range(4, cout + 3, 4):
                    for fb in [8, 4, 2]:
                        for wb in [8, 4, 2]:
                            #if i != 4 or j != 4:
                            #    continue
                            if cin == 3:
                                continue
                            count += 1
                            
                            if lists[16] != 1:
                                if i != j: # assume same input/output for depth-wise conv
                                    continue
                                else:
                                    group = i

                            result.add("width_{}-height_{}-cin_{}-cout_{}-kernel_{}-stride_{}-pad_{}-group_{}-fb_{}-wb_{}-base".format(
                                width, height, i, j, kernel, stride, "SAME", group, fb, wb))
                            result.add("width_{}-height_{}-cin_{}-cout_{}-kernel_{}-stride_{}-pad_{}-group_{}-fb_{}-wb_{}-layer".format(
                                width, height, i, j, kernel, stride, "SAME", group, fb, wb))
            print(line, count)
            total_count += count
        print("Total configuration number (contains redundant) %d" % total_count)

def load_list(record="models/list.txt", result=set()):
    count = len(result)
    with open(record) as f:
        lines = f.readlines()
        f.close()
        for line in lines:
            if line[0] == "#":
                continue
            if '#' in line:
                line = line.split('#')[0]
            lists = obtainInt(line.split())
            if len(lists) == 0:
                continue
            assert len(lists) == 8, 'unexpected length %s' % line
            iteration_type = lists[0]
            width = lists[1]
            height = lists[2]
            cin = lists[3]
            cout = lists[4]
            kernel = lists[5] 
            stride = lists[6]
            group = lists[7]

            layer_count = 0
            if iteration_type == 0: # iterate output channel
                for i in range(4, cout + 3 , 4):
                    for fb in [8, 4, 2]:
                        for wb in [8, 4, 2]:
                            assert group == 1, "Unexpect"
                            result.add("width_{}-height_{}-cin_{}-cout_{}-kernel_{}-stride_{}-pad_{}-group_{}-fb_{}-wb_{}-base".format(
                                width, height, cin, i, kernel, stride, "SAME", group, fb, wb))
                            result.add("width_{}-height_{}-cin_{}-cout_{}-kernel_{}-stride_{}-pad_{}-group_{}-fb_{}-wb_{}-layer".format(
                                width, height, cin, i, kernel, stride, "SAME", group, fb, wb))
                            layer_count += 2

            if iteration_type == 1: # iterate input channel
                for i in range(4, cin + 3 , 4):
                    for fb in [8, 4, 2]:
                        for wb in [8, 4, 2]:
                            assert group == 1, "Unexpect"
                            result.add("width_{}-height_{}-cin_{}-cout_{}-kernel_{}-stride_{}-pad_{}-group_{}-fb_{}-wb_{}-base".format(
                                width, height, i, cout, kernel, stride, "SAME", group, fb, wb))
                            result.add("width_{}-height_{}-cin_{}-cout_{}-kernel_{}-stride_{}-pad_{}-group_{}-fb_{}-wb_{}-layer".format(
                                width, height, i, cout, kernel, stride, "SAME", group, fb, wb))
                            layer_count += 2

            if iteration_type == 2: # iterate bit config only
                for fb in [8, 4, 2]:
                    for wb in [8, 4, 2]:
                        result.add("width_{}-height_{}-cin_{}-cout_{}-kernel_{}-stride_{}-pad_{}-group_{}-fb_{}-wb_{}-base".format(
                            width, height, cin, cout, kernel, stride, "SAME", group, fb, wb))
                        result.add("width_{}-height_{}-cin_{}-cout_{}-kernel_{}-stride_{}-pad_{}-group_{}-fb_{}-wb_{}-layer".format(
                            width, height, cin, cout, kernel, stride, "SAME", group, fb, wb))
                        layer_count += 2

            if iteration_type == 3: # iterate input/output channel
                for i in range(4, cin + 3 , 4):
                    for j in range(4, cout + 3, 4):
                        for fb in [8, 4, 2]:
                            for wb in [8, 4, 2]:
                                assert group == 1, "Unexpect"
                                result.add("width_{}-height_{}-cin_{}-cout_{}-kernel_{}-stride_{}-pad_{}-group_{}-fb_{}-wb_{}-base".format(
                                    width, height, i, j, kernel, stride, "SAME", group, fb, wb))
                                result.add("width_{}-height_{}-cin_{}-cout_{}-kernel_{}-stride_{}-pad_{}-group_{}-fb_{}-wb_{}-layer".format(
                                    width, height, i, j, kernel, stride, "SAME", group, fb, wb))
                                layer_count += 2

            if iteration_type == 4: # iterate input/output channel
                cin_step = max(cin // 64, 4)
                for i in range(cin_step, cin + cin_step - 1, cin_step):
                    cout_step = max(cout // 64, 4)
                    for j in range(cout_step, cout + cout_step - 1, cout_step):
                        for fb in [8, 4, 2]:
                            for wb in [8, 4, 2]:
                                if lists[7] != 1:
                                    if i != j:
                                        continue
                                    else:
                                        group = i
                                result.add("width_{}-height_{}-cin_{}-cout_{}-kernel_{}-stride_{}-pad_{}-group_{}-fb_{}-wb_{}-base".format(
                                    width, height, i, j, kernel, stride, "SAME", group, fb, wb))
                                result.add("width_{}-height_{}-cin_{}-cout_{}-kernel_{}-stride_{}-pad_{}-group_{}-fb_{}-wb_{}-layer".format(
                                    width, height, i, j, kernel, stride, "SAME", group, fb, wb))
                                layer_count += 2
            print("Layer variant count %d" % layer_count)
    print("Configuration number: before {}, new added {}, current {}".format(count, len(result)-count, len(result)))
    return result


def parase_bench_name(bench_name):
    hyparamater = dict()
    items = bench_name.split('-')
    for i in items:
        i = i.split('_')
        if len(i) != 2:
            continue
        hyparamater[i[0]] = i[1]
    return hyparamater

def get_bench_nn(bench_name, WRPN=False):

    g = Graph(bench_name, dataset='nop', log_level=logging.INFO)
    batch_size = 16
    hyparamater = parase_bench_name(bench_name)
    width       = hyparamater['width']
    width       = int(width)
    height      = hyparamater['height']
    height      = int(height)
    cin         = hyparamater['cin']
    cin         = int(cin)
    cout        = hyparamater['cout']
    cout        = int(cout)
    kernel_size = hyparamater['kernel']
    kernel_size = int(kernel_size)
    stride      = hyparamater['stride']
    stride      = (1, int(stride), int(stride), 1)
    pad         = hyparamater['pad'].upper()
    group       = hyparamater['group']
    group       = int(group)
    fb          = hyparamater['fb']
    wb          = hyparamater['wb']
    fb          = {'8': FQDtype.FXP8, '4': FQDtype.FXP4, '2': FQDtype.FXP2}[fb]
    wb          = {'8': FQDtype.FXP8, '4': FQDtype.FXP4, '2': FQDtype.FXP2}[wb]

    with g.as_default():
        with g.name_scope('inputs'):
            prev = get_tensor(shape=(batch_size, width, height, cin), name='data', dtype=FQDtype.FXP8, trainable=False)

        with g.name_scope('conv1'):  # prepare
            prev = conv(prev, filters=cin, kernel_size=1, stride=(1,1,1,1), pad='SAME', c_dtype=FQDtype.FXP8, w_dtype=FQDtype.FXP8)

        if 'base' in bench_name:
            return g

        with g.name_scope('conv2'):  # profile
            prev = conv(prev, filters=cout, kernel_size=kernel_size, stride=stride, pad=pad, group=group, c_dtype=fb, w_dtype=wb)

    return g

def save_list(lists=None, filename=None):
    assert len(lists) % 2 == 0, "list length should be even, but found %d" % len(lists)

bucket = 50
benchlist = []

def load_config():
    index = os.getenv('bitfusion_index')
    try:
        index = int(index)
    except:
        index = None
    print("Index", index)

    result = set()
    for root, dirnames, filenames in os.walk('models'):
        #print(root, dirnames, filenames)
        if root != 'models':
            continue
        for files in filenames:
            if '.txt' not in files:
                continue
            record = os.path.join(root, files)
            print("Processing file %s" % record)
            result = load_list(record=record, result=result)
            #break
    # reduce already profile one
    # convert to list and partition
    benchlist = list(result)
    benchlist.sort()

    interval = len(benchlist) // bucket
    if interval % 2 != 0:
        interval = interval - 1
    print('interval is being set to %d' % interval)
    return benchlist, index, interval

if __name__ == "__main__":
    bench, index, interval = load_config()
    for i in range(bucket + 1):
        if i*interval < len(bench):
            if (i+1)*interval <= len(bench):
                benchlist = bench[i*interval: i*interval + interval]
            else:
                benchlist = bench[i*interval:]
        else:
            benchlist = []
        print("benchlist with index %d length: %d" % (i, len(benchlist)))
else:
    bench, index, interval = load_config()
    if index is not None:
        i = index
        if i*interval < len(bench):
            if (i+1)*interval <= len(bench):
                benchlist = bench[i*interval: i*interval + interval]
            else:
                benchlist = bench[i*interval:]
        else:
            benchlist = []
        print("benchlist with index %d length: %d" % (i, len(benchlist)))



