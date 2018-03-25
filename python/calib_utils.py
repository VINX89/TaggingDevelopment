#! /usr/bin/env python
# coding: utf-8 

"""
@file     calib_utils.py
@author   Kevin Heinicke, Vincenzo Battista
@date     17/09/2017

@brief    Collection of routines useful for calibration purposes
"""

import logging
log = logging.getLogger(__name__)

def calibration_curve(y_true, y_prob, bins=10, sample_weight=None):
    """
    Calculate a calibration curve for probability values

    @param y_true: column of true target
    @param y_prob: column of predicted classifier probability
    @param bins: number of bins
    @param sample_weight: column of weights to assign (e.g. sWeights)
    """

    import numpy as np
    from sklearn.utils import column_or_1d

    if sample_weight is None:
        sample_weight = np.ones_like(y_true)

    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)

    if type(bins) is int:
        bins = np.linspace(0, 1, bins)
    else:
        bins = bins[:-1]

    binids = np.digitize(y_prob, bins)

    if sample_weight is not None:
        bin_sums = np.bincount(binids, weights=sample_weight * y_prob, minlength=len(bins))
        bin_true = np.bincount(binids, weights=sample_weight * y_true, minlength=len(bins))
        bin_total = np.bincount(binids, weights=sample_weight, minlength=len(bins))
    else:
        bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
        bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
        bin_total = np.bincount(binids, minlength=len(bins))
    
    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    log.info(f'bin_sums: {bin_sums}')
    log.info(f'bin_true: {bin_true}')
    log.info(f'bin_total: {bin_total}')
    log.info(f'prob_true = bin_true/bin_total: {prob_true}')
    log.info(f'prob_pred = bin_sums/bin_total: {prob_pred}')
        
    return prob_true, prob_pred, (bin_sums[nonzero], bin_true[nonzero], bin_total[nonzero])


def bootstrap_calibrate_prob(labels, weights, probs, n_calibrations=30,
                             threshold=0., symmetrize=False):
    """
    Bootstrap isotonic calibration (borrowed from tata-antares/tagging_LHCb):
    * randomly divide data into train-test
    * on train isotonic is fitted and applyed to test
    * on test using calibrated probs p(B+) D2 and auc are calculated
    
    :param probs: probabilities, numpy.array of shape [n_samples]
    :param labels: numpy.array of shape [n_samples] with labels
    :param weights: numpy.array of shape [n_samples]
    :param threshold: float, to set labels 0/1j
    :param symmetrize: bool, do symmetric calibration, ex. for B+, B-
    
    :return: D2 array and auc array
    """

    import numpy as np
    from sklearn.isotonic import IsotonicRegression
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import roc_auc_score

    aucs = []
    D2_array = []
    labels = (labels > threshold) * 1

    for _ in range(n_calibrations):
        (train_probs, test_probs,
         train_labels, test_labels,
         train_weights, test_weights) = train_test_split(
            probs, labels, weights, train_size=0.5)
        iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        if symmetrize:
            iso_reg.fit(np.r_[train_probs, 1-train_probs],
                        np.r_[train_labels > 0, train_labels <= 0],
                        np.r_[train_weights, train_weights])
        else:
            iso_reg.fit(train_probs, train_labels, train_weights)
            
        probs_calib = iso_reg.transform(test_probs)
        alpha = (1 - 2 * probs_calib) ** 2
        aucs.append(roc_auc_score(test_labels, test_probs,
                                  sample_weight=test_weights))
        D2_array.append(np.average(alpha, weights=test_weights))
    return np.array(D2_array), np.array(aucs)
