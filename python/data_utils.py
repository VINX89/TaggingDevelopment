#! /usr/bin/env python
# coding: utf-8

"""
@file      data_utils.py
@author    Kevin Heinicke, Vincenzo Battista
@date      17/09/2017

@brief     Collection of utilities for data handling
"""

import warnings
import logging
from itertools import islice

from tqdm import tqdm

import numpy as np
import ROOT
import pandas as pd
from root_pandas import read_root

log = logging.getLogger(__name__)

def concat_df_chunks(filenames, chunksize, **kwargs):

    from itertools import chain

    return chain(
        *(read_root(f, chunksize=chunksize, **kwargs) for f in filenames)
        )


class NSplit(object):
    """
    Create object to handle data frame splitting
    and iterations over splitted data frames
    """
    def __init__(self, df, splits=3, seed=42, shuffle=True, unique_columns=['eventNumber', 'runNumber', 'nCandidate']):
        self.df = df
        np.random.seed(seed)
        # only drop column in it is already an integer column
        self.df.reset_index(inplace=True,
                            drop=not any(col in unique_columns for col in self.df.index.names))
        self.unique_events = self.df.groupby(unique_columns)[
            self.df.columns[0]].idxmax()
        self.raw_indices = self.df.index.values
        if shuffle:
            np.random.shuffle(self.unique_events)
        if type(splits) is list:
            #ok, let's forget about duck typing for a second
            #divide data into len(splits)+1 chunks, each of them containing a different fraction of samples
            ncand = self.unique_events.sum()
            print(f"ncand={ncand}")
            cand_split = []
            for frac in splits:
                cand_splits.append( int( float(ncand)*frac ) )
                print(f"cand_splits={int( float(ncand)*frac )}")
            self.raw_index_sets = np.array_split(self.unique_events, cand_splits)
        else:
            #divide data in N chunks of equal size
            self.raw_index_sets = np.array_split(self.unique_events, splits)

    def __iter__(self):
        for index_set in self.raw_index_sets:
            yield self.raw_indices[self.df.index.isin(index_set)]

def create_train_test_calib(df, seed=42, cand_splits=None, unique_columns=['eventNumber', 'runNumber', 'nCandidate']):
    """
    Split data frame randomly in three subsets.

    @param df: input dataframe
    @param seed: seed for random splitting
    """
    splits=3
    if type(cand_splits) is list and len(cand_splits)>2:
        raise ValueError("The list of fractions has to contains no more than 2 elements")
    df_sets = [df.iloc[indices] for indices in NSplit(df, splits=splits, seed=seed, unique_columns=unique_columns)]
    return df_sets[0], df_sets[1], df_sets[2]

def split3fold(df, splits=[0.33,0.33], seed=42, unique_columns=['eventNumber', 'runNumber', 'nCandidate']):

    np.random.seed(seed)
    data = df.reset_index(inplace=False, drop=not any(col in unique_columns for col in df.index.names))
    unique_events = data.groupby(unique_columns)[ data.columns[0] ].idxmax()
    n_events = unique_events.sum() 
    raw_indices = data.index.values
    ids = np.arange(n_events)
    np.random.shuffle( ids )
    

def get_event_number(df=None, weight_column='',
                     config=None, files=None, preselection=None, **pandas_kwargs):
    """ Compute the total number of events contained in a given tuple. The
    tuple can either directly be passed into this funciton or it will be
    read from a file list given via files or a configuration file.

    Parameters
    ----------
    df : a dataframe containing the signal yield column
    weight_column : name of the signal yield weight column
    config : configuration dictionary
    expected to contain the keys
    - 'filepath'
    - 'files'
    - 'pandas_kwargs'
    files : list of root files used to read the dataframe if df is not given
    pandas_kwargs : pandas kwargs passed to read_root if df is not given
    """
    dfExists = df is not None
    if dfExists and df.empty:
        log.warning("Empty dataframe")
    # first perform some sanity checks and warn the user if the parameter
    # combination will not work
    if not any([dfExists, config, files]):
        raise NameError("You need to define a dataframe, a list of files or a"
                        " config object.")
    if df is not None and files:
        warnings.warn("files will be ignored if df is given")
    if df is not None and config:
        warnings.warn("config will be ignored if df is not None")

    # merge config and pandas kwargs
    if not config:
        config = {
            'pandas_kwargs': pandas_kwargs,
        }

    # get the df
    if not dfExists:
        if not files:
            files = [config['filepath'] + f for f in config['files']]
        cols=['nCandidate']
        if weight_column != '':
            cols += [weight_column]    
        log.debug('create data frame from file')
        df = read_root(files, key=config['pandas_kwargs']['key'],
                       columns=cols)

    # apply preselection, if required
    if preselection:
        df = df.query(preselection)
        
    if weight_column != '':
        # use unique events if the df has been flattened
        if '__array_index' in df.columns or '__array_index' in df.index.names:
            res = df.groupby(['runNumber', 'eventNumber', 'nCandidate'])[weight_column].first().sum()
            log.debug('array index defined')
        else:
            log.debug('no array index defined')
            res = df[df.nCandidate == 0][weight_column].sum()
    else:
        if '__array_index' in df.columns or '__array_index' in df.index.names:
            res = df.groupby(['runNumber', 'eventNumber', 'nCandidate']).ngroups
            log.debug('array index defined')
        else:
            log.debug('no array index defined')
            res = df[df.nCandidate == 0].shape[0]
            
    log.debug(f'number of events: {res}')
    return res

def read_and_split(file_in_path,
                   tree_in,
                   data_kwargs,
                   query='',
                   splits = None,
                   weight_column='',
                   maxslices=None,
                   chunksize=5000):

    import ROOT
    import pandas as pd
    from root_pandas import read_root
    from tqdm import tqdm
    from itertools import islice
    
    rootfile = ROOT.TFile(file_in_path)
    tree = rootfile.Get(tree_in)
    entries = tree.GetEntries()
    if maxslices is not None and maxslices < entries/chunksize:
        total = maxslices
    else:
        total = entries/chunksize

    N_evt = 0
    df_list = []

    if splits:
        N_evt_split = []
        df_split_list = []
        df_split_selected = []
        for split in splits:
            N_evt_split.append(0)
            df_split_list.append( [] )
            df_split_selected.append( pd.DataFrame() )

    for df_unselected in tqdm(islice(read_root(file_in_path, **data_kwargs), maxslices), total=total):

        #Count all candidates, select and merge
        N_evt += get_event_number(df_unselected)
        if query != '':
            df_selected = df_unselected.query(query)
        else:
            df_selected = df_unselected
        df_list.append( df_selected )
        df_selected = pd.concat(df_list)

        if splits:
            for s, split in enumerate(splits):
                df_split_unselected = df_unselected.query(split)
                N_evt_split[s] += get_event_number(df_split_unselected)

                if query != '':
                    df_split_selected[s] = df_split_unselected.query(query)
                else:
                    df_split_selected[s] = df_split_unselected

                df_split_list[s].append( df_split_selected[s] )
                df_split_selected[s] = pd.concat(df_split_list[s])

    if query != '':
        log.info(f'Total events before selection: {N_evt}')
        if splits:
            for s, split in enumerate(splits):
                log.info(f'...split: {split}, events {N_evt_split[s]}')
        log.info(f'Total events after selection: {get_event_number(df_selected, weight_column)}')
        if splits:
            for s, split in enumerate(splits):
                log.info(f'...split: {split}, events {get_event_number(df_split_selected[s], weight_column)}')
    else:
        log.info(f'Total events: {N_evt}')
        if splits:
            log.info(f'...split: {split}, events {N_evt_split[s]}')

    if splits:
        return N_evt, N_evt_split, df_selected, df_split_selected
    else:
        return N_evt, df_selected
    
def read_train_test(file_in_path,
                    tree_in,
                    data_kwargs,
                    query='',
                    weight_column='SigYield_sw',
                    b_id='B_ID',
                    tagp_id='B_OSMuonDev_TagPartsFeature_ID',
                    maxslices=None,
                    chunksize=5000,
                    do_split=True):
    """
    Create train and test dataframe from input ROOT file.
    The splitting is made according to the event number.
    A target is created automatically by comparing the sign
    of B candidate and tagging particle

    @param file_in_path: input ROOT file
    @param tree_in: TTree object name in ROOT file
    @param data_kwargs: additional arguments for the root_pandas 'read_root' function
    @param query: preliminary selection
    @param weight_column: name of the signal yield weight column
    @param b_id: name of B candidate ID leaf in ROOT file
    @param tagp_id: name of tagging particle ID leaf in ROOT file
    @param maxslices: maximum number of 'slices' to divide data into (to avoid 100% RAM consumption)
    @param chunksize: size of each chunk to divide data into (to avoid 100% RAM consumption)
    @param debug: print some debug info
    @param do_split: do train/test splitting
    """

    import ROOT
    import pandas as pd
    from tqdm import tqdm
    from itertools import islice

    rootfile = ROOT.TFile(file_in_path)
    tree = rootfile.Get(tree_in)
    entries = tree.GetEntries()
    if maxslices is not None and maxslices < entries/chunksize:
        total = maxslices
    else:
        total = entries/chunksize

    df_list = []
    df_train_list = []
    df_test_list = []
    N_evt = 0
    N_evt_train = 0
    N_evt_test = 0
    
    for df_unselected in tqdm(islice(read_root(file_in_path, **data_kwargs), maxslices), total=total):

        #Add useful columns
        if do_split:
            df_unselected['random_split'] = (df_unselected['runNumber']+df_unselected['eventNumber'])%2==0
        df_unselected['target'] = np.sign(df_unselected[b_id]) == np.sign(df_unselected[tagp_id])

        #Count all candidates, select and merge
        N_evt += get_event_number(df_unselected)
        if query != '':
            df_selected = df_unselected.query(query)
        else:
            df_selected = df_unselected
        df_list.append( df_selected )
        df_selected = pd.concat(df_list)

        #Do random train/test splitting as well
        if do_split:
            df_train_unselected = df_unselected.query('random_split==True')
            df_test_unselected = df_unselected.query('random_split==False')
            #Count train/test candidates
            N_evt_train += get_event_number(df_train_unselected)
            N_evt_test += get_event_number(df_test_unselected)
            if query != '':
                #Apply preliminary selection
                df_train_selected = df_train_unselected.query(query)
                df_test_selected = df_test_unselected.query(query)
            else:
                df_train_selected = df_train_unselected
                df_test_selected = df_test_unselected
            #Append to lists
            df_train_list.append( df_train_selected )
            df_test_list.append( df_test_selected )
            
            df_train_selected = pd.concat(df_train_list)
            df_test_selected = pd.concat(df_test_list)

    if query != '':
        log.info(f'Total events before selection: {N_evt}')
        if do_split:
            log.info(f'...train sample: {N_evt_train}')
            log.info(f'...test sample: {N_evt_test}')
        log.info(f'Total events after selection: {get_event_number(df_selected, weight_column)}')
        if do_split:
            log.info(f'...train sample: {get_event_number(df_train_selected, weight_column)}')
            log.info(f'...test sample: {get_event_number(df_test_selected, weight_column)}')
    else:
        log.info(f'Total events: {N_evt}')
        if do_split:
            log.info(f'...train sample: {N_evt_train}')
            log.info(f'...test sample: {N_evt_test}')

    if do_split:
        return N_evt, N_evt_train, N_evt_test, df_selected, df_train_selected, df_test_selected
    else:
        return N_evt, df_selected


def apply_selection(df, selections):
    """Apply a query of selections.
    Essentially a shortcut for df.query(' and '.join(selections)), preventing
    some weird bugs.
    """
    # TODO: this ensures that queries dont use more than maxQ features
    #       there seems to be a bug otherwise. With later pandas
    #       versions this should be dropped and replaced by a simple
    #       long query.
    maxQ = 10
    for s in [selections[i * maxQ:i * maxQ + maxQ]
              for i in range(int(len(selections) / maxQ) + 1)]:
        if len(s):
            df.query(' and '.join(s), inplace=True)
    return df


def filter_event_particles(df, event_cols, sorting_feature, nMax):
    """Group df by event_cols and select nMax particles per event, ordered by
    sorting_feature.
    """
    grouped = df.groupby(event_cols, sort=False)
    # calculate indices of the top n rows in each group;
    # depending on how many particles are found in each group, the index
    # needs to be reset. This seems to be a bug and might be fixed in 0.20
    # see pandas github issue #15297
    try:
        if grouped[sorting_feature].count().max() > nMax:
            indices = grouped[sorting_feature].nlargest(nMax).reset_index([0, 1]).index
        else:
            indices = grouped[sorting_feature].nlargest(nMax).index
    except ValueError:
        print(f'A pandas error has been ignored while reading {tqdm().n}th'
              'slice of the current input file: {e}')

    return df.loc[np.unique(indices.values)]

def load_data_fast(file_in, index=['runNumber', 'eventNumber', 'nCandidate', '__array_index']):

    from root_pandas import read_root
    df = read_root(file_in)
    df.set_index(index, inplace=True)
    return df


def load_data(files=None, chunksize=None, max_chunks=None, index_features=None,
              unique_event_features=None, selections=None,
              sorting_feature=None, particles_per_event=None, config={},
              **kwargs):
    """Read a dataframe from given file or list of files.

    Parameters
    ----------
    files : list or str
        List of filenames or single filename to read.
    chunksize : int
        Process the files in chunks of `chunksize` events. A progressbar
        will indicate the read progress.
    max_chunks : int
        If `chunksize` is defined, only `max_chunks` of the data will be read.
    index_features : list of str
        Defines which features within the ROOT files will be used to uniquely
        identify single tagging particles.
    unique_event_features : list of str
        Defines which features are used to group tagging particles by event.
        Usually this should be `['runNumber', 'eventNumber', 'nCandidate']`.
    selections : list of str
        Apply a set of selections while reading in data. Each line will be
        joined with ` and `. Also see `pandas.DataFrame.query()` for details.
    sorting_feature : str
        A single feature column which will be used to sort tagging particles
        within every event.
    particles_per_event : int
        Define how many particles should be kept per event, if a
        sorting_feature is given.
    config : dict
        Dictionary containing relevant configuration. Mainly for compatibility
        with json config files. Every key in `config` will be overwritten by
        kwargs that were explicitly passed to this function.
    kwargs : dict
        These will be passed to `pandas.read_root`.
    """
    # first, some argument logics and sanity checks
    # read dataset, create a list of absolute filenames
    if 'pandas_kwargs' in config:
        pd_kwargs = config['pandas_kwargs']
        pd_kwargs.update(kwargs)
    else:
        pd_kwargs = kwargs

    # finally update with this functions kwargs if applicable
    for key in ['chunksize']:
        if locals()[key]: pd_kwargs[key] = locals()[key]

    # files should be present in config if not passed by argument
    if not files:
        files = [os.path.join(config.get('filepath', ''), f)
                 for f in config['files']]
    elif type(files) is str:
        files = [files]

    # quickly get the number of events
    print('Reading ', end='')
    entries = 0
    for f in files:
        rootfile = ROOT.TFile(f)
        tree = rootfile.Get(pd_kwargs.get('key'))
        entries += tree.GetEntriesFast()
    chunksize = chunksize or pd_kwargs['chunksize']
    # depending on chunksize and max_chunks, get the number of events that will
    # actually be read
    total = (max_chunks
             if max_chunks is not None and max_chunks < (entries / chunksize)
             else (entries / chunksize))

    print(total * chunksize, 'events.', flush=True)

    merged_training_df = None

    index_cols = index_features or config.get('index_features')
    event_cols = unique_event_features or config.get('unique_event_features')

    # loop over tuple and fill training variables
    for df in tqdm(
            islice(read_root(files, **pd_kwargs), max_chunks),
            total=total):
        # set a proper index
        df.set_index(index_cols, inplace=True, drop=True)

        # apply selections
        selections = selections or config.get('selections')
        if selections:
            selected_df = apply_selection(df, selections)
        else:
            selected_df = df

        # select n max pt particles
        sorting_feature = sorting_feature or config.get('sorting_feature')
        if sorting_feature:
            max_d = filter_event_particles(df, event_cols, sorting_feature)
        else:
            max_df = selected_df

        # append this chunk to the training dataframe
        merged_training_df = pd.concat([merged_training_df, max_df])

    return merged_training_df


def tot_event_number(base_file, base_tree, weight_column='', unique_events=['runNumber', 'eventNumber', 'nCandidate'], preselection=None, presel_column=[]):
    columns = []
    if weight_column != '':
        columns.append( weight_column )
    for pr in presel_column:
        columns.append( pr )
    for ue in unique_events:
        columns.append( ue )
    df = read_root(base_file, key=base_tree, columns=columns)
    if preselection is not None:
        df = df.query( preselection )
    if weight_column != '':
        nevt = df.groupby(unique_events)[weight_column].head(1).sum()
    else:
        nevt = df.groupby(unique_events).ngroups
    return nevt

def sel_efficiency(df, total_event_number, weight_column='', unique_events=['runNumber', 'eventNumber', 'nCandidate']):

    from uncertainties import ufloat

    # efficiency, using binomial uncertainty on the number of selected candidates
    if weight_column != '':
        if not unique_events:
            selection_efficiency = df[weight_column].sum() / total_event_number
        else:
            selection_efficiency = df.groupby(unique_events)[weight_column].head(1).sum() / total_event_number
    else:
        if not unique_events:
            selection_efficiency = len(df) / total_event_number
        else:
            selection_efficiency = df.groupby(unique_events).ngroups / total_event_number
    
    selection_efficiency = ufloat(
        selection_efficiency,
        np.sqrt(selection_efficiency * (1 - selection_efficiency) * total_event_number) / total_event_number
        )

    log.info('ε = {:.3u}%'.format(selection_efficiency*100))

    return selection_efficiency
