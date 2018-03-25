#! /usr/bin/env python
# coding: utf-8 

"""
@file    plot_utils.py
@author  Vincenzo Battista
@date    16/09/2017

@brief   Collection of utilities to simplify plotting
"""

import logging
log = logging.getLogger(__name__)

def plot_roc_auc(df, prob_col="probas", target_col="target", lhcb_label=None):
    """
    """
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve
    
    score = "{0:.2f}".format( roc_auc_score(df[target_col], df[prob_col]) )
    fpr, tpr, _ = roc_curve(df[target_col], df[prob_col])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(fpr,
             tpr,
             lw=1,
             label=f"ROC (AUC={score})")
    plt.plot([0,1], [0,1], lw=2, linestyle="--", label="random tagger")
    ax.set_xlabel('False Positive Rate', ha='right', x=1)
    ax.set_ylabel('True Positive Rate', ha='right', y=1)
    ax.legend(loc='best')
    
    if lhcb_label is not None:
        ax.text(0.2,
                0.1,
                lhcb_label["label"],
                {'size': lhcb_label["size"]})
        ax.minorticks_on()
            
    plt.minorticks_on()
    plt.tight_layout()
    
    return fig
    
def plot_correlation(df, mva_dict, weight_column='SigYield_sw'):
    """
    WARNING: numbers are computed with sWeights, but colors are not.
    This is fair enough unless sWeighted data has very different
    correlations for some reason
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm as cm
    from stat_utils import weighted_correlation
    import numpy as np
    
    data = df[list(mva_dict['Features'].keys()) + [weight_column] ]
    corr_matrix = df[list(mva_dict['Features'].keys())].corr()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(corr_matrix, 
                     interpolation="nearest", 
                     cmap=cmap)
    labels = [ mva_dict['Features'][feat]['label'] for feat in list(mva_dict['Features'].keys()) ]
    jump_x = corr_matrix.shape[0]*1.0/len(labels)
    jump_y = corr_matrix.shape[1]*1.0/len(labels)

    ax1.set_xticklabels(labels,fontsize=16, rotation='vertical')
    ax1.set_yticklabels(labels,fontsize=16)
    ax1.set_xticks(np.arange(0,corr_matrix.shape[0], jump_x))
    ax1.set_yticks(np.arange(0,corr_matrix.shape[1], jump_y))
    ax1.tick_params('both', length=0, width=0, which='major')
    ax1.tick_params('both', length=0, width=0, which='minor')
    ax1.set_aspect('auto')
    fig.colorbar(cax)
    
    x_positions = np.linspace(start=0, stop=len(labels), num=len(labels), endpoint=False)
    y_positions = np.linspace(start=0, stop=len(labels), num=len(labels), endpoint=False)
    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            x_feat = list(mva_dict['Features'].keys())[x_index]
            y_feat = list(mva_dict['Features'].keys())[y_index]
            c = "{0:.2f}".format(weighted_correlation(data[x_feat], 
                                                      data[y_feat],
                                                      data[weight_column]))
            ax1.text(x, y, str(c), {'size': 12}, color='black', ha='center', va='center')

    return fig
    
def plot_features(plot_dict, df, figsize=(30,30), columns=2, weight_column='', lhcb_label=None):
    """
    Plots dataframe df for the two possible values of the target.

    @param plot_dict: dictionary with plot settings. It requires
                      a structure like the one provided in the following
                      example:

                      plot_dict = {
                          'Features' : { 'df_feature_colname_1' : {'bins': 80, 'log': True, 'label': '$f_{1}$ [MeV/c]'},
                                         'df_feature_colname_2' : {'bins': 20, 'log': False, 'label': '$f_{2}$ [mm]'} },
                          'Target'   : { 'target==0' : 'Wrong Tag',
                                         'target==1' : 'Right Tag' } 
                           }

    @param df: pandas dataframe. The column names have to match the plot_dict['Features'] keys,
               and the plot_dict['Target'] keys have to be valid queries for this data frame.

    @param figsize: 2d tuple containing x and y figure size

    @param columns: number of columns of the plot grid

    @param weight_column: name of weight column in the data frame

    @param lhcb_label: dictionary to set up LHCb label.
                       The following structure is required:
                       lhcb_label = {
                          'x'     : x,
                          'y'     : y,
                          'label' : 'LHCb',
                          'size'  : 28}

    """
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    mva_features = list(plot_dict["Features"].keys())
    rows = int(len(mva_features)/columns) if len(mva_features)%columns==0 else int((len(mva_features)/columns)+1)
    
    fig, axs = plt.subplots(rows, columns, figsize=figsize)
    idx=0
    for row in range(0,rows):
        for col in range(0,columns):
            if len(mva_features)%columns!=0 and idx==len(mva_features):
                #Delete last empty box and stop
                fig.delaxes(axs[row,col])
                break
            feat = mva_features[idx]
            target0 = list(plot_dict["Target"].keys())[0]
            target1 = list(plot_dict["Target"].keys())[1]
            ax = axs[row,col]
            _ = ax.hist(df.query(target1)[feat],
                        color='blue', 
                        bins=plot_dict["Features"][feat]["bins"], 
                        alpha=0.3, 
                        normed=True,
                        weights=None if weight_column == '' else df.query(target1)[weight_column].values)
            _ = ax.hist(df.query(target0)[feat],
                        color='red', 
                        bins=plot_dict["Features"][feat]["bins"], 
                        alpha=0.3, 
                        normed=True,
                        weights=None if weight_column == '' else df.query(target0)[weight_column].values)
            if plot_dict["Features"][feat]["log"]:
                ax.set_yscale('log')
            ax.set_xlabel(plot_dict["Features"][feat]["label"], ha='right', x=1)
            ax.set_ylabel('Entries (a.u.)', ha='right', y=1)
            patch0 = mpatches.Patch(color='blue',
                                    label=plot_dict["Target"][target0],
                                    alpha=0.3)
            patch1 = mpatches.Patch(color='red',
                                    label=plot_dict["Target"][target1],
                                    alpha=0.3)
            ax.legend(loc='best', handles=[patch0,patch1])
            if lhcb_label is not None:
                ax.text(lhcb_label["x"],
                        lhcb_label["y"],
                        lhcb_label["label"],
                        {'size': lhcb_label["size"]})
            ax.minorticks_on()
            if df.query(target1)[feat].max()>100000 or df.query(target0)[feat].max()>100000:
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                ax.get_xaxis().get_offset_text().set_position((0.1,0))
                
            idx += 1
    plt.minorticks_on()
    plt.tight_layout()
    
    return fig

def plot_XGB_importance(booster, plot_dict, ax=None, height=0.2,
                        xlim=None, ylim=None, title='Feature importance',
                        xlabel='F score', ylabel='Features', max_num_features=None,
                        grid=True, lhcb_label=None, **kwargs):
    """
    Plots feature importance for a trained XGB classifier.
    The importance is the so called F-score, i.e. the total number of
    times (occurrences) that a particolar feature has been used in a split
    node across all the trees of the ensemble.

    This function has been reimplemented from the xgboost library:
    https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/plotting.py

    @param booster: the XGB classifier
    @param input dictionary: (see plot_features documentation)
    @param ax: input matplotlib subplot
    @param height: height of the produced plot
    @param xlim: x-axis limit
    @param ylim: y-axis limit
    @param title: plot title
    @param xlabel: x-axis title
    @param ylabel: y-axis title
    @param max_num_features: maximum number of features
    @param grid: add grid to the plot
    @param lhcb_label: dictionary to set up LHCb label.
                       The following structure is required:
                       lhcb_label = {
                          'x'     : x,
                          'y'     : y,
                          'label' : 'LHCb',
                          'size'  : 28} 
    @param **kwargs: list of additional drawing options for the histogram (ax.barh) 
    """

    import matplotlib.pyplot as plt
    from xgboost import XGBClassifier
    from xgboost.core import Booster
    from xgboost.sklearn import XGBModel
    import numpy as np

    importance = booster.booster().get_fscore()

    if len(importance) == 0:
        raise ValueError('Booster.get_score() results in empty')

    tuples = [(k, importance[k]) for k in importance]
    if max_num_features is not None:
        tuples = sorted(tuples, key=lambda x: x[1])[-max_num_features:]
    else:
        tuples = sorted(tuples, key=lambda x: x[1])

    features, values = zip(*tuples)
    labels = []
    for feat in features:
        labels.append( plot_dict["Features"][feat]['label'] )

    if ax is None:
        fig, ax = plt.subplots(1, 1)
        
    ylocs = np.arange(len(values))
    ax.barh(ylocs, values, align='center', height=height, **kwargs)

    for x, y in zip(values, ylocs):
        ax.text(x + 1, y, x, va='center', fontsize=14)

    ax.set_yticks(ylocs)
    ax.set_yticklabels(labels)

    if xlim is not None:
        if not isinstance(xlim, tuple) or len(xlim) != 2:
            raise ValueError('xlim must be a tuple of 2 elements')

    else:
        xlim = (0, max(values) * 1.1)
    ax.set_xlim(xlim)

    if ylim is not None:
        if not isinstance(ylim, tuple) or len(ylim) != 2:
            raise ValueError('ylim must be a tuple of 2 elements')
    else:
        ylim = (-1, len(values))
    ax.set_ylim(ylim)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel, ha='right', x=1)
    if ylabel is not None:
        ax.set_ylabel(ylabel, ha='right', y=1)

    if lhcb_label is not None:
        ax.text(lhcb_label["x"],
                lhcb_label["y"],
                lhcb_label["label"],
                {'size': lhcb_label["size"]})

    ax.minorticks_on()
    plt.minorticks_on()
    ax.grid(grid)
    
    return fig

def plot_classifier_output(df_train,
                           df_test,
                           nbins=50,
                           xrange=None,
                           yrange=None,
                           ks_dict=None,
                           target_dict=None,
                           type="probas",
                           title="Predicted probability",
                           weight_column='',
                           lhcb_label=None):
    """
    Plots predicted output of a classifier.
    It requires train and test data frames as input.
    It is assumed that either 'probas' or 'etas' columns exist in these data frame.
    On the plot, the result of a KS test is reported in order to check the compatibility between the distributions
    (they should be compatible in order to avoid overtraining).

    @param df_train: train data frame
    @param df_test: test data frame
    @param nbins: number of bins for x-axis
    @param xrange: tuple with (min,max) for x-axis (leave None to pickup default)
    @param yrange: tuple with (min,max) for y-axis (leave None to pickup default)
    @param ks_dict: dictionary with x-y coordinates to print result of KS test
                    for compatibility between train and test samples.
                    E.g.:
                    ks_dict = {'x': 0.31, 'y': 5.}
                    If None (default), nothing is printed
    @param target_dict: dictionary with target definition.
                        E.g:
                        target_dict = {'target==0': 'Wrong Tag', 'target==1': 'Right Tag'}
                        keys have to be valid pandas queries.
                        If None (default), all targets are merged into a single histogram
    @param x_txt: x position for left edge of text box
    @param y_txt: y position of upper edge of text box
    @param type: quantity to plot ('probas': classifier probability. 'etas': probability converted to predicted mistag. Add 'calib_' prefix to take calibrated values)
    @param title: plot title
    @param weight_column: name of weight column in the data frame
    @param lhcb_label: dictionary to set up LHCb label.
                           The following structure is required:
                           lhcb_label = {
                           'x'     : x,
                           'y'     : y,
                           'label' : 'LHCb',
                           'size'  : 28} 
    """

    if type not in ["etas","probas","calib_etas","calib_probas"]:
        raise ValueError("ERROR: 'type' argument of 'plot_classifier_output' is not 'etas', 'probas', 'calib_etas' or 'calib_probas'")

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy.stats import ks_2samp

    fig = plt.figure()
    ax = fig.add_subplot(111)

    handles = []

    if type=="probas":
        plt.xlabel('P(correct)', ha='right', x=1)
        text = '$<P>$'
    elif type=="calib_probas":
        plt.xlabel('calibrated P(correct)', ha='right', x=1)
        text = '$<P>$'
    elif type=="etas":
        plt.xlabel(r'$\eta$', ha='right', x=1)
        text = '$<\eta>$'
    elif type=="calib_etas":
        plt.xlabel(r'$\omega$', ha='right', x=1)
        text = '$<\omega>$'
        
    if target_dict:
        #Split distributions per target value
        for target, color, hatch, alpha, fill in zip(list(target_dict.keys()), ['blue', 'red'], [None, "/"], [0.6, 1.0], [True, False]):
            #Plot histogram with bars
            df_train.query(target)[type].hist(ax=ax,
                                              color=color,
                                              edgecolor=color,
                                              bins=nbins,
                                              alpha=alpha,
                                              normed=True,
                                              fill=fill,
                                              hatch=hatch,
                                              weights=None if weight_column=='' else df_train.query(target)[weight_column].values,
                                              label=f'Train sample, {target_dict[target]}',
                                              range=xrange if xrange else None)
            #Plot histogram as points with Poisson error
            ptest = hist_errorbars(ax,
                                   df_test.query(target)[type],
                                   color=color,
                                   ecolor=color,
                                   bins=nbins,
                                   alpha=alpha,
                                   normed=True,
                                   weights=None if weight_column=='' else df_test.query(target)[weight_column].values,
                                   labelstring=f'Test sample, {target_dict[target]}',
                                   range=xrange if xrange else None)

    else:
        df_train[type].hist(ax=ax,
                            color='blue',
                            bins=nbins,
                            alpha=0.3,
                            normed=True,
                            label='Train sample',
                            weights=None if weight_column=='' else df_train[weight_column].values,
                            range=xrange if xrange else None)
        df_test[type].hist(ax=ax,
                           color='red',
                           bins=nbins,
                           alpha=0.3,
                           normed=True,
                           label='Test sample',
                           weights=None if weight_column=='' else df_test[weight_column].values,
                           range=xrange if xrange else None)
            
    plt.ylabel('Entries (a.u.)', ha='right', y=1)

    if ks_dict:
        #Print KS test result
        y_shift=0.0
        for target in list(target_dict.keys()):
            
            K, p = ks_2samp(df_train.query(target)[type].values, 
                            df_test.query(target)[type].values)
            x_txt = ks_dict['x']
            y_txt = ks_dict['y']
            
            if weight_column == '':
                mean_train = df_train.query(target)[type].sum() / len( df_train.query(target) )
                mean_test = df_test.query(target)[type].sum() / len( df_test.query(target) )
            else:
                mean_train = (df_train.query(target)[type] * df_train.query(target)[weight_column]).sum() / df_train.query(target)[weight_column].sum()
            
                mean_test = (df_test.query(target)[type] * df_test.query(target)[weight_column]).sum() / df_test.query(target)[weight_column].sum()
            
            plt.text(x_txt,
                     y_txt-y_shift,
                     f'{target_dict[target]}',
                     {'size': 18})
            plt.text(x_txt,
                     y_txt-0.3-y_shift, 
                     r'Train '+text+'={:.3f}'.format(mean_train),
                     {'size': 18})
            plt.text(x_txt,
                     y_txt-0.6-y_shift,
                     r'Test '+text+'={:.3f}'.format(mean_test),
                     {'size': 18})
            plt.text(x_txt,
                     y_txt-0.9-y_shift,
                     r'KS={:.3f}, p-value={:.3f}'.format(K, p),
                     {'size': 18})
            y_shift += 1.4

    ax.legend(loc='best')
    ax.minorticks_on()
    ax.grid(b=False)

    if lhcb_label is not None:
        ax.text(lhcb_label["x"],
                lhcb_label["y"],
                lhcb_label["label"],
                {'size': lhcb_label["size"]})

    plt.title(title)
    plt.minorticks_on()
    
    if yrange:
        plt.ylim(*yrange)
    
    plt.grid(b=False)
    
    return fig

def plot_calib_curve(df, sample_weight, calib_model, title="Calibration curves", calib_label="calibration", type="probas", x_lim=[0.3,0.9], y_lim=[0.3,0.9], lhcb_label=None):
    """
    Plot ideal calibration curve, uncalibrated probability (or mistag), calibrated probability (or mistag) and calibration model

    @param df: data frame. It must contain the following columns: target, probas (or etas), calib_probas (or calib_etas)
    @param sample_weight: per-event weight column (e.g. sWeights)
    @param calib_model: calibration model used to fit data
    @param title: plot title
    @param calib_label: label to tag calibration model
    @param type: either 'probas' or 'etas'
    @param x_lim: x-asis limits (array)
    @param y_lim: y-axis limits (array)
    @param lhcb_label: dictionary to set up LHCb label.
                       The following structure is required:
                       lhcb_label = {
                       'x'     : x,
                       'y'     : y,
                       'label' : 'LHCb',
                       'size'  : 28}
    """

    if type not in ["etas","probas"]:
        raise ValueError("ERROR: 'type' argument of 'plot_classifier_output' is neither 'etas' nor 'probas'")

    import matplotlib.pyplot as plt
    import calib_utils
    from calib_utils import calibration_curve
    import numpy as np

    fig = plt.figure()    
    
    xs = np.linspace(0, 1)
    ys = calib_model.predict_proba(xs)[:, 1]

    if type == "probas":
        target = df.target
        prob = df.probas
        calib_prob = df.calib_probas
        interval = [0, 1]
        x_label = 'predicted P(correct)'
        y_label = 'true P(correct)'
    else:
        target = ~df.target
        prob = df.etas
        calib_prob = df.calib_etas
        interval = [0, 0.52]
        xs = 1 - xs
        ys = 1 - ys
        x_label = r'$\eta$'
        y_label = r'$\omega$'

    # first, plot the calibration curve and the "perfect calibration" line
    plt.plot(interval, interval, '--', label='perfect calibration')
    plt.plot(xs, ys, label=calib_label)

    # plot the uncalibrated data
    bins = np.percentile(prob, np.linspace(0, 100, 8))
    log.info(f'binning: {bins}')
    
    prob_true, prob_pred, (bin_sums, bin_true, bin_total) = calibration_curve(target, prob, sample_weight=sample_weight, bins=bins)
    xerrs = [prob_pred - bins[:-1], bins[1:] - prob_pred]
    yerrs = np.sqrt(prob_true * (1 - prob_true) * bin_total) / bin_total
    plt.errorbar(prob_pred, prob_true, yerrs, xerrs, '.', label='uncalibrated')
        
    # plot the calibrated data
    log.debug(f'calib prob (or eta) min: {calib_prob.min()}')
    log.debug(f'calib prob (or eta) max: {calib_prob.max()}')
    
    if bins.min() > calib_prob.min():
        bins[0] = calib_prob.min()
    if bins.max() < calib_prob.max():
        bins[-1] = calib_prob.max()
        
    prob_true, prob_pred, (bin_sums, bin_true, bin_total) = calibration_curve(target, calib_prob, sample_weight=sample_weight, bins=bins)
    xerrs = [prob_pred - bins[:-1], bins[1:] - prob_pred]
    yerrs = np.sqrt(prob_true * (1 - prob_true) * bin_total) / bin_total
    
    log.debug(f'xerrs: {xerrs}')
    log.debug(f'yerrs: {yerrs}')
    
    plt.errorbar(prob_pred, prob_true, yerrs, xerrs, '.', label='calibrated')

    # fine-tune plot
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.xlabel(x_label, ha='right', x=1)
    plt.ylabel(y_label, ha='right', y=1)
    plt.title(title)
    plt.legend(loc='best')
    if lhcb_label is not None:
        plt.text(lhcb_label["x"],
                 lhcb_label["y"],
                 lhcb_label["label"],
                 {'size': lhcb_label["size"]})
    plt.minorticks_on()
    
    return fig

def plot_partial_dependency(df_train, clf, mva_dictionary, weights, figsize=(40,60), columns=2, type="probas", logx=True):
    """
    Plot classifier output (probability or mistag) as a function of each input feature while
    marginalizing over the remaining N-1 features.
    This is the so called importance.

    @param df_train: the training dataset. It has to contain the classifier output 'etas' or 'probas'

    @param clf: the trained classifier.

    @param mva_dictionary: a dictionary with the following, minimal structure (example):
    
                           plot_dict = {
                             'Features' : { 'df_feature_colname_1' : {'label': '$f_{1}$ [MeV/c]'},
                                            'df_feature_colname_2' : {'label': '$f_{2}$ [mm]'} }

                           Each 'df_feature_colname_{}' sub-dictionary can contain an additional min/max
                           key, which determine the plotting range for that feature

    @param figsize: 2d tuple containing x and y figure size

    @param columns: number of columns of the plot grid

    @param type: either 'probas' or 'etas' (predicted probability or mistag)

    @param logx: plot horizontal axis (i.e. feature value) in log scale (recommendend for better readibility)
    """

    if type not in ["etas","probas"]:
        raise ValueError("ERROR: 'type' argument of 'plot_partial_dependency' is neither 'etas' nor 'probas'")
    
    import tagging_utils
    from mva_utils import partial_dependency
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    feature_names = list(mva_dictionary["Features"].keys())

    probs = df_train[type]
    X_train = df_train[list(mva_dictionary['Features'].keys())]
    
    rows = int(len(feature_names)/columns) if len(feature_names)%columns==0 else int((len(feature_names)+1)/columns)
    
    fig, axs = plt.subplots(rows, columns, figsize=figsize)
    idx=0
    for row in range(0,rows):
        for col in range(0,columns):
            if len(feature_names)%columns!=0 and idx==len(feature_names):
                #Delete last empty box and stop
                fig.delaxes(axs[row,col])
                break
            
            f = feature_names[idx]
            f_id = X_train.columns.tolist().index(f)
            
            log.info(f"Plotting {f}...")
            
            grid, p_pred = partial_dependency(clf,
                                              X_train,
                                              weights,
                                              f_id,
                                              type)
            
            ax = axs[row,col]
            label = "Predicted Probability"
            y_bins = np.linspace(0, 1, 50)
            if type == "etas":
                label = "Predicted Mistag"
                y_bins = np.linspace(0, 0.5, 50)
            _, _, _, cax = ax.hist2d(X_train.values[:, X_train.columns.tolist().index(f)],
                                     probs,
                                     bins=(grid, y_bins),
                                     weights=weights,
                                     alpha=0.9,
                                     cmap=plt.cm.Blues,
                                     normed=True)
            #ax.plot(X_train.values[:, X_train.columns.tolist().index(f)],
            #        probs,
            #        'o',
            #        color = 'blue',
            #        alpha = 0.1,
            #        label=label)
            ax.plot(grid,
                    p_pred,
                    '-',
                    color = 'red',
                    linewidth = 5.0,
                    label='Average')
            
            if "min" in list(mva_dictionary["Features"][f].keys()):
                min_x = mva_dictionary["Features"][f]["min"]
            else:
                min_x = min(grid)
                
            if "max" in list(mva_dictionary["Features"][f].keys()):
                max_x = mva_dictionary["Features"][f]["max"]
            else:
                max_x = max(grid)
                
            ax.set_xlim(min_x, max_x)
            ax.set_xlabel(mva_dictionary["Features"][f]["label"])
            ax.set_ylabel('$\eta$')

            #prob_patch = mpatches.Patch(color='blue', label=label, alpha=1.0)
            #avg_patch = mpatches.Patch(color='red', label='Average', alpha=1.0)
            #ax.legend(loc='best', handles=[prob_patch,avg_patch])

            if logx:
                ax.set_xscale('log')
            
            #cbar = plt.colorbar(cax)
            
            idx += 1
            
    plt.minorticks_on()
    plt.tight_layout()

    return fig

def hist_errorbars(ax, data, xerrs=True, labelstring='', *args, **kwargs) :
    """
    Plot a histogram with error bars. Accepts any kwarg accepted by either numpy.histogram or pyplot.errorbar
    Adapted from:
    https://gist.github.com/neggert/2399228
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import inspect
    
    # pop off normed kwarg, since we want to handle it specially
    norm = False
    if 'normed' in list(kwargs.keys()) :
        norm = kwargs.pop('normed')
        
    # retrieve the kwargs for numpy.histogram
    histkwargs = {}
    for key, value in kwargs.items() :
        if key in inspect.getargspec(np.histogram).args :
            histkwargs[key] = value
            
    histvals, binedges = np.histogram( data, **histkwargs )

    yerrs = np.sqrt(histvals)
    
    if norm :
        nevents = float(sum(histvals))
        binwidth = (binedges[1]-binedges[0])
        histvals = histvals/nevents/binwidth
        yerrs = yerrs/nevents/binwidth
        
    bincenters = (binedges[1:]+binedges[:-1])/2
    
    if xerrs :
        xerrs = (binedges[1]-binedges[0])/2
    else :
        xerrs = None

    ebkwargs = {}
    for key, value in kwargs.items() :
        if key in inspect.getargspec(plt.errorbar).args :
            ebkwargs[key] = value
    plot = ax.errorbar(bincenters, histvals, yerr=yerrs, xerr=xerrs, label=labelstring, fmt=".", **ebkwargs)    
    
    if 'log' in list(kwargs.keys()) :
        if kwargs['log'] :
            ax.set_yscale('log')
            
    if 'range' in list(kwargs.keys()) :
        ax.set_xlim(*kwargs['range'])

    return plot
