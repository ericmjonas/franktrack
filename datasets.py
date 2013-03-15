import glob
import os
import cPickle as pickle

CURRENT_EPOCHS = [
    'bukowski_05.W1', 
    'bukowski_02.W1', 
    'bukowski_02.C', 
    'bukowski_04.W1', 
    'bukowski_04.W2',
    'bukowski_01.linear', 
    'bukowski_01.W1', 
    'bukowski_01.C', 
    'bukowski_05.linear', 
    'Cummings_03.w2', 
    'Cummings_03.linear', 
    'Dickinson_02.c', 
    'H206.2', 
    'Dickinson_01.w1', 
    'Cummings_05.linear', 
    'Cummings_06.c',
    'Cummings_06.w1'
          ]

HARD_EPOCHS = ['Dickinson_01.w1', 
             'Cummings_05.linear', 
             'Cummings_08.linear', 
             'Cummings_06.c', 
             'Cummings_08.linear', 
             'Bukowski_02.W2', 
             'Dickinson_04.c', 
             'Cummings_06.w1']

CURRENT_FRAMES = [0, 500, 1000]

def all():
    EPOCHS = [os.path.basename(f) for f in glob.glob("data/fl/*")]
    for epoch in EPOCHS:
        for frame in CURRENT_FRAMES:
            yield epoch, frame

def bad():
    return pickle.load(open('currentset.pickle', 'r'))['bad_epochs']

def current():
    return [('Cummings_07.w2', 1000), 
            ('Cummings_08.linear', 0), 
            ('I106.4', 0), 
            ('Cummings_06.w2', 500)]
