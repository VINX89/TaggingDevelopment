#! /usr/bin/env python
# coding: utf-8

"""
@file     mva_utils.py
@author   Kevin Heiniche, Vincenzo Battista
@date     31/10/2017

@brief    Collection of utilities for optimisation of
          MVA classifier used for Flavour Tagging
"""

import logging
log = logging.getLogger(__name__)

def xgb_cv_tuning(df,
                  n_estimators,
                  max_depth,
                  tot_event_number,
                  max_column,
                  mva_features,
                  outdir = None,
                  sample_weight=None,
                  nBootstrap=10,
                  njobs=24,
                  learning_rate=0.01,
                  lhcb_label = None,
                  cand_index = ['runNumber', 'eventNumber', 'nCandidate']):
    """
    """

    import numpy as np
    from xgboost import XGBClassifier
    import matplotlib
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from data_utils import NSplit, create_train_test_calib
    from calib_models import PolynomialLogisticRegression
    from tagging_utils import tagging_power_score
    from sklearn.metrics import roc_auc_score, roc_curve
    from scipy import stats
    import math
    
    plot_dict = {}
    roc_dict = {}
    fpr_mean = np.linspace(0, 1, 100)
    
    log.info(f"train BDT over the following features:")
    print(mva_features)

    #We evaluate tagging power on 1/3 of the sample...
    nTot = float(tot_event_number)/3.0

    #loop over possible maximum depths
    for mdep in max_depth:

        tp_mean = []
        tp_err = []
        roc_auc_mean = []
        roc_auc_err = []

        plot_dict[str(mdep)] = {}
        roc_dict[str(mdep)] = {}

        #loop over possible number of estimators
        for idx, nest in enumerate(n_estimators):

            roc_dict[str(mdep)][str(nest)] = {}

            roc_auc_scores = []
            fprs = []
            tprs = []
            tag_powers = []
            
            for bst in range(nBootstrap):

                # yield 3-fold split for CV
                df_sets = create_train_test_calib(df, seed=int(42+bst+mdep+nest))

                for i in range(3):
                    df1, df2, df3 = (df_sets[i % 3].copy(),
                                     df_sets[(i + 1) % 3].copy(),
                                     df_sets[(i + 2) % 3].copy())

                    #train on df1
                    model = XGBClassifier(nthread=njobs,
                                          n_estimators=nest,
                                          max_depth=mdep,
                                          learning_rate=learning_rate,
                                          seed=int(42+bst+mdep+nest))
                    model.fit(df1[mva_features], df1.target, sample_weight=df1[sample_weight])
                    
                    #calibrate df2
                    df2['probas'] = model.predict_proba(df2[mva_features])[:, 1]
                    calibrator = PolynomialLogisticRegression(power=2,
                                                              solver='lbfgs',
                                                              n_jobs=njobs)
                    calibrator.fit(df2.probas.values.reshape(-1, 1), df2.target, sample_weight=df2[sample_weight])
                
                    #evaluate performance on df3 (taking particle which maximises "max_column", e.g. pT)
                    df3['probas'] = model.predict_proba(df3[mva_features])[:, 1]
                    df3['calib_probas'] = calibrator.predict_proba(df3.probas.values.reshape(-1,1))[:, 1]
                    df3['calib_etas'] = np.where(df3.calib_probas > 0.5, 1 - df3.calib_probas, df3.calib_probas)
                    max_pt_particles = df3.loc[df3.groupby(cand_index)[max_column].idxmax()]
                    tag_power = tagging_power_score(max_pt_particles.calib_etas,
                                                    tot_event_number=nTot,
                                                    sample_weight=max_pt_particles[sample_weight])
                    tag_powers.append( tag_power )
                    
                    #get roc auc as well
                    roc_auc_scores.append( roc_auc_score(df3['target'], df3['calib_probas']) )
                    fpr, tpr, _ = roc_curve(df3['target'], df3['calib_probas'])
                    tprs.append(np.interp(fpr_mean, fpr, tpr).tolist())

            #Compute average performance for this mdep/nest pair
            tag_powers = np.array( tag_powers )
            tp_mean.append( np.mean( tag_powers ) )
            tp_err.append( stats.sem( tag_powers ) )
            roc_auc_scores = np.array( roc_auc_scores )
            roc_auc_mean.append( np.mean( roc_auc_scores ) )
            roc_auc_err.append( stats.sem( roc_auc_scores ) )
            log.info(f'''average performance for maximum_depth={mdep} and n_estimators={nest}:
            tagging power = {tp_mean[idx]*100} +/- {tp_err[idx]*100} %
            roc auc = {roc_auc_mean[idx]*100} +/- {roc_auc_err[idx]*100} %
            ''')

            #Build "average" roc curve as well
            #Fix here: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py ???
            tprs = np.array( tprs )

            roc_dict[str(mdep)][str(nest)]["tpr_mean"] = tprs.mean(axis=0)

        tp_mean = np.array( tp_mean )
        tp_err = np.array( tp_err )     
        roc_auc_mean = np.array( roc_auc_mean )
        roc_auc_err = np.array( roc_auc_err )
        
        plot_dict[str(mdep)]["tp_mean"] = tp_mean
        plot_dict[str(mdep)]["tp_err"] = tp_err
        plot_dict[str(mdep)]["tp_max"] = np.max( tp_mean ) #highest tp within this maximum_depth
        plot_dict[str(mdep)]["best_tp_nest"] = n_estimators[ np.argmax( tp_mean ) ] #n_estimators giving best tp within this maximum_depth

        plot_dict[str(mdep)]["roc_auc_mean"] = roc_auc_mean
        plot_dict[str(mdep)]["roc_auc_err"] = roc_auc_err
        plot_dict[str(mdep)]["roc_auc_max"] = np.max( roc_auc_mean ) #highest roc auc within this maximum_depth
        plot_dict[str(mdep)]["best_roc_auc_nest"] = n_estimators[ np.argmax( roc_auc_mean ) ] #n_estimators giving best roc auc within this maximum_depth

    log.info("grid search completed!!!")

    tp_list = []
    nest_list = []
    mdep_list = list(plot_dict.keys())
    for mdep in mdep_list:
        tp_list.append( plot_dict[mdep]["tp_max"] )
    tp_list = np.array( tp_list )
    max_tp = np.max( tp_list )
    best_tp_mdep = int(mdep_list[ np.argmax( tp_list ) ])
    best_tp_nest = int(plot_dict[str(best_tp_mdep)]["best_tp_nest"])
    log.info(f"best tagging power: {max_tp*100} %")
    log.info(f"...max_depth: {best_tp_mdep}")
    log.info(f"...n_estimators: {best_tp_nest}")

    roc_auc_list = []
    nest_list = []
    mdep_list = list(plot_dict.keys())
    for mdep in mdep_list:
        roc_auc_list.append( plot_dict[mdep]["roc_auc_max"] )
    roc_auc_list = np.array( roc_auc_list )
    max_roc_auc = np.max( roc_auc_list )
    best_roc_auc_mdep = int(mdep_list[ np.argmax( roc_auc_list ) ])
    best_roc_auc_nest = int(plot_dict[str(best_roc_auc_mdep)]["best_roc_auc_nest"])
    log.info(f"best roc auc: {max_roc_auc*100} %")
    log.info(f"...max_depth: {best_roc_auc_mdep}")
    log.info(f"...n_estimators: {best_roc_auc_nest}")

    if outdir:

        figs = []
        axs = []

        #Plot tagging power and roc auc first
        for i, fom, title in zip(range(2),
                                 ["tp", "roc_auc"],
                                 ['Per-event tagging power (%)', 'ROC AUC (%)']):
            figs.append( plt.figure() )
            axs.append( figs[i].add_subplot(111) )

            n_est = np.array( n_estimators )
        
            for mdep in list(plot_dict.keys()):
                plt.errorbar( n_est,
                              100*plot_dict[mdep][f"{fom}_mean"],
                              100*plot_dict[mdep][f"{fom}_err"],
                              label = f"max depth = {mdep}" )

            axs[i].set_xlabel('Number of trees', ha='right', x=1)
            axs[i].set_ylabel(title, ha='right', y=1)

            axs[i].legend(loc='best')

            if lhcb_label is not None:
                axs[i].text(lhcb_label["x"][fom],
                            lhcb_label["y"][fom],
                            lhcb_label["label"],
                            {'size': lhcb_label["size"]})
            axs[i].minorticks_on()

            plt.minorticks_on()
            plt.tight_layout()
            plt.savefig(outdir+f'crossval_{fom}.pdf')

        #Plot roc curves as well
        figs.append( plt.figure() )
        axs.append( figs[2].add_subplot(111) )

        for mdep in list(roc_dict.keys()):
            for nest in list(roc_dict[mdep].keys()):

                plt.plot(fpr_mean,
                         roc_dict[mdep][nest]["tpr_mean"],
                         lw=1,
                         label=f"md={mdep}, nt={nest}")

        plt.plot([0,1], [0,1], lw=2, linestyle="--", label="random")
                
        axs[2].set_xlabel('False Positive Rate', ha='right', x=1)
        axs[2].set_ylabel('True Positive Rate', ha='right', y=1)

        axs[2].legend(loc='best')

        if lhcb_label is not None:
            axs[2].text(0.2,
                        0.1,
                        lhcb_label["label"],
                        {'size': lhcb_label["size"]})
        axs[2].minorticks_on()
            
        plt.minorticks_on()
        plt.tight_layout()
        plt.savefig(outdir+f'roc_curve.pdf')

    return best_tp_mdep, best_tp_nest, best_roc_auc_mdep, best_roc_auc_nest
    
    
def partial_dependency(clf, X_train, weights, f_id, type="probas"):
    """
    Calculate the dependency (or partial dependency) of a response variable on a predictor (or multiple predictors)
    1. Sample a grid of values of a predictor.
    2. For each value, replace every row of that predictor with this value, calculate the average prediction.

    Adapted from:
    https://xiaoxiaowang87.github.io/monotonicity_constraint/

    @param clf: the fitted classifier (a xgboost instance)
    @param X_train: the training data (the very same object passed to xgboost for training)
    @param f_id: index of the feature to be plotted withing the input data
    @param type: either 'probas' or 'etas' (return classifier probability or mistag)
    """

    import pandas as pd
    import numpy as np
    from stat_utils import weighted_percentile

    if type not in ["probas", "etas"]:
        log.warning("type argument assumed to be 'probas'")
        type = "probas"

    feature_ids = [X_train.columns.tolist().index(f) for f in X_train.columns]
    
    X_temp = X_train.copy().values
    col_temp = X_train.columns.tolist()
    
    grid = np.linspace(weighted_percentile(X_temp[:, f_id], 0.1, weights),
                        weighted_percentile(X_temp[:, f_id], 99.9, weights),
                        50)
    #grid = np.linspace(np.percentile(X_temp[:, f_id], 0.1),
    #                   np.percentile(X_temp[:, f_id], 99.9),
    #                   50)
    p_pred = np.zeros(len(grid))
    
    if len(feature_ids) == 0 or f_id == -1:
        log.error('input error!')
        return None, None
    else:
        for i, val in enumerate(grid):
            
            X_temp[:, f_id] = val
            data = pd.DataFrame(data=X_temp[:, feature_ids].reshape( (len(X_temp), len(feature_ids)) ), columns=col_temp)
            p_pred[i] = np.average(clf.predict_proba(data)[:, 1], weights=weights)
            if type == "etas":
                p_pred[i] = np.where(p_pred[i] > 0.5, 1 - p_pred[i], p_pred[i])
            
    return grid, p_pred  
