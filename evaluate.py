from ruffus import *
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
    algorithms = ['current']
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

    confs = compare.avg_delta_conf_threshold(delta, algo_data['confidence'], 
                                               tholds)
    print "tholds=", tholds
    print "confs=", confs
    pickle.dump({'confs': confs, 
                 'tholds' : tholds,
                 'dataset' : dataset, 
                 'algorithm' : algorithm}, 
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

# @files(agg_comparisons, "comparisons.html")
# def generate_comparison_html(blah):
#     """
#     All datasets, all algorithms
#     """


if __name__ == "__main__":
    pipeline_run([run_comparison, agg_comparisons])
