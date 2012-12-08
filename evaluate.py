from ruffus import *
from matplotlib import pylab
import numpy as np
import os
import pandas
from glob import glob
import track
import compare
import cPickle as pickle

# for each epoch
# get the data
# compare
# write

DATA_DIR = "data/fl"
def ddir(x):
    return os.path.join(DATA_DIR, x)

REPORT_DIR = "results"
def rdir(x):
    return os.path.join(REPORT_DIR, x)

def comparisons():
    # for each dataset 
    datasets = glob(ddir("*"))
    
    # for each algorithm 
    algorithms = ['current', 'centroid']
    for dataset in datasets:
        for algorithm in algorithms:
            dataset_name = dataset[len(DATA_DIR)+1:]
            
            truth_file = rdir(os.path.join(dataset_name, "truth.npy"))
            algo_file = rdir(os.path.join(dataset_name, "algo", 
                                          algorithm + ".npy"))
            comparison_file = algo_file + ".comparison.pickle"
            yield ([truth_file, algo_file], # inputs
                   comparison_file, # output
                   dataset_name, algorithm)
        # run the comparison, generate the file

@follows(mkdir(os.path.join(REPORT_DIR, "comparisons")))
@files(comparisons)
def run_comparison((truth_file, algorithm_output), 
                   comparison_filename, dataset, algorithm):
    truth_data = np.load(truth_file)
    algo_data = np.load(algorithm_output)
    
    delta = compare.xy_compare(algo_data, truth_data)
    tholds = np.arange(0, 1.0, 0.1)

    confs, fractions = compare.avg_delta_conf_threshold(delta, 
                                                        algo_data['confidence'], 
                                                        tholds)
    deltas_at_thold = []
    for thold_i, thold in enumerate(tholds):
        idx = np.argwhere(algo_data['confidence'] >= thold)
        tholded_deltas = delta[idx]

        deltas_at_thold.append(tholded_deltas)

    print "tholds=", tholds
    print "confs=", confs
    pickle.dump({'errors_at_conf': confs, 
                 'fractions': fractions,
                 'tholds' : tholds,
                 'dataset' : dataset, 
                 'algorithm' : algorithm, 
                 'deltas_at_thold': deltas_at_thold}, 
                open(comparison_filename, 'w'))

@merge(run_comparison, 'comparisons.pickle')
def agg_comparisons(inputfiles, outputfile):
    """
    Create dataframe of comparison results

    """
    dfs = []
    for f in inputfiles:
        d = pickle.load(open(f, 'r'))
        # crete the data frame
        dfs.append(pandas.DataFrame(d))
    print dfs
    df_all = pandas.concat(dfs, ignore_index=True)
    pickle.dump(df_all, open(outputfile, 'w'))


@files(agg_comparisons, "comparisons.html")
def generate_comparison_html(inputfile, outputfile):
    """
    All datasets, all algorithms
    """
    comparisons = pickle.load(open(inputfile, 'r'))
    html = comparisons.to_html()
    open(outputfile, 'w').write(html)

THOLD = 0.9
@merge(run_comparison, 'deltas_at_thold.%02.2f.pickle' % THOLD)
def agg_deltas_at_thold(inputfiles, outputfile):
    """
    """
    dfs = []
    for f in inputfiles:
        d = pickle.load(open(f, 'r'))
        tholds = d['tholds']
        thold_i = np.argwhere(tholds == THOLD)
        delta = d['deltas_at_thold'][thold_i].flatten()
        dataset = d['dataset']
        algorithm = d['algorithm']
        print "Algorithm = ", algorithm, delta.shape
        df = pandas.DataFrame({'algorithm' : algorithm, 
                               'dataset' : dataset, 
                               'fraction' : float(d['fractions'][thold_i]), 
                               'delta' : delta, 
                               })
        dfs.append(df)

    df = pandas.concat(dfs, ignore_index=True)
    
    pickle.dump(df, open(outputfile, 'w'))

@files(agg_deltas_at_thold, ['output.pdf'])
def plot_box_whisker(inputfile, outputfiles):
    df = pickle.load(open(inputfile, 'r'))
    
    for algo in ['centroid', 'current']:
        f = pylab.figure(figsize=(16, 4))
        ax = f.add_subplot(1, 1, 1)
        df_algo = df[df['algorithm'] == algo]
        groups  = df_algo.groupby('dataset')
        raw_deltas = []
        ticks = []
        fractions = []
        for name, group in groups:
            raw_deltas.append(group['delta'])
            ticks.append(name)
            print group['fraction']
            fractions.append(np.mean(group['fraction']))
        ax.boxplot(raw_deltas, positions=range(len(ticks)))
        ax.set_xticklabels(ticks, rotation=90, 
                  size='x-small')
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(range(len(ticks)))
        ax2.set_xticklabels(["%2.0f" % (fr*100) for fr in fractions])
            
        #df_algo.boxplot(column=['delta'], by=['dataset'], ax=ax)
        f.savefig('figs/deltas.%s.png' % algo, dpi=300)

if __name__ == "__main__":
    pipeline_run([run_comparison, agg_comparisons, agg_deltas_at_thold, 
                  plot_box_whisker
                  ])
