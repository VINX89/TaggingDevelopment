#! /usr/bin/env python
# coding: utf-8 

"""
@file     tagging_utils.py
@author   Kevin Heiniche, Vincenzo Battista
@date     17/09/2017

@brief    Collection of utilities for flavour tagging
"""
import numpy as np


import logging
log = logging.getLogger(__name__)

def d2_score(y_score, sample_weight=None):
    """ Compute <D^2> = <(1 - 2*omega)^2> where omega is either the per-event
    mistag estimate or the per event probability of the tag being correct.
    
    Parameters
    ----------
    y_score : array-like, shape=(n_samples,)
    omega or p(correct) values
    
    sample_weight : array-like, shape=(n_samples,), optional, default: None
    Weights. If set to None, all weights will be set to 1
    
    Returns
    -------
    score : float
    D squared
    """

    import numpy as np
    
    if sample_weight is None:
        sample_weight = np.ones_like(y_score)
    D2s = (1 - 2 * y_score)**2

    # This seems to return an unexpected nan value from time to time
    # return np.average(D2s, weights=sample_weight)
    return np.sum(sample_weight * D2s) / np.sum(sample_weight)


def tagging_power_score(y_score, efficiency=None, tot_event_number=None,
                        sample_weight=None):
    """ Compute per event tagging power with selection efficiency
    
    Parameters
    ----------
    y_score : array-like, shape=(n_samples,)
    omega or p(correct) values
    
    efficiency : float, optional, default: None
    the selection efficiency
    
    tot_event_number : float, optional, default: None
    the total number of events (tagged and untagged)
    
    sample_weight : array-like, shape=(n_samples,), optional, default: None
    Weights. If set to None, all weights will be set to 1
    
    Returns
    -------
    score : float
    tagging power
    """

    import numpy as np

    if sample_weight is None:
        sample_weight = np.ones_like(y_score)

    if efficiency is not None and tot_event_number is None:
        return efficiency * d2_score(y_score, sample_weight)
    if tot_event_number is not None and efficiency is None:
        return 1 / tot_event_number * np.sum(sample_weight
                                             * (1 - 2 * y_score)**2)
    else:
        raise NotImplementedError("Either efficiency or tot_event_number must be passed!")

def calc_tagfracs_per_cand(df, weight_column='SigYield_sw'):
    """
    For each B candidate, compute the fraction of right and wrong tag
    particles among all the tagging particles associated to that B

    @param df: input dataframe
    @param weight_column: name of the signal yield weight column 
    """
    
    import pandas as pd
    import numpy as np

    # group by candidates
    if '__array_index' in df.columns or '__array_index' in df.index.names:
        df_grouped = df.groupby(['runNumber', 'eventNumber', 'nCandidate'])
    else:
        df_grouped = df[df.nCandidate == 0]

    # create new df
    df_tagfrac_per_cand = pd.DataFrame({
        'N_TagParts': df_grouped['target'].size(),
        'N_R_TagParts': df_grouped['target'].sum(),
        weight_column: df_grouped[weight_column].first()
        })
    
    df_tagfrac_per_cand['f_R'] = np.divide(df_tagfrac_per_cand['N_R_TagParts'],df_tagfrac_per_cand['N_TagParts'])
    df_tagfrac_per_cand['f_W'] = 1 - df_tagfrac_per_cand['f_R']
    
    return df_tagfrac_per_cand


def calculate_tagging_performance(df_candidates,
                                  num_sig_cands,
                                  weight_column='SigYield_sw',
                                  pt_name='B_OSMuonDev_TagPartsFeature_PT',
                                  print_result=False,
                                  take_max_pt=False):
    """
    Compute average tagging efficiency, mistag fraction and tagging power.

    @param df_candidates: dataframe with input B candidates (passing an input selection)
    @param num_sig_cands: normalisation, i.e. number of events before any cut is applied
    @param weight_column: name of the signal yield weight column
    @param pt_name: name of tagging particle pT column in df_candidates (to be set only if max pT rule is used)
    @param print_result: print some output info
    @param take_max_pt: if True, compute average mistag taking only one tagging particle per candidate (the one with max pT).
                        if False, compute average mistag from average number of wrong tag particles across all B candidates.
    """

    import math
    import numpy as np
    from uncertainties import ufloat

    if df_candidates.size == 0:
        return (0., 0.5, 0.)

    if take_max_pt:
        df_candidates.reset_index(drop=True, inplace=True)
        df_candidates = df_candidates.iloc[df_candidates.groupby(['runNumber', 'eventNumber', 'nCandidate'])[pt_name].idxmax()]
        if '__array_index' in df_candidates.columns or '__array_index' in df_candidates.index.names:
            N_tagged = np.sum(df_candidates.groupby(['runNumber', 'eventNumber', 'nCandidate'])[weight_column].first())
        else:
            N_tagged = np.sum(df_candidates[df_candidates.nCandidate == 0][weight_column].first())
        tag_omega = np.sum(df_candidates[weight_column] * ~df_candidates.target) / np.sum(df_candidates[weight_column])
    else:
        N_tagged = df_candidates[weight_column].sum()
        tag_omega = (df_candidates[weight_column] * df_candidates.f_W).sum() / (df_candidates[weight_column] * (df_candidates.f_R + df_candidates.f_W)).sum()

    tag_eff = N_tagged/num_sig_cands
    tag_eff_err_sq = (tag_eff*(1.-tag_eff)/num_sig_cands)
    
    tag_eff_err = 0.
    if tag_eff_err_sq >= 0. and tag_eff_err_sq <= 1.:
        tag_eff_err = math.sqrt(tag_eff_err_sq)
        
    tag_eff_w_u = ufloat(tag_eff, tag_eff_err)

    tag_omega_err_sq = tag_omega*(1.-tag_omega)/N_tagged
    tag_omega_err = 0.
    if tag_omega_err >= 0. and tag_omega_err <= 1.:
        try:
            tag_omega_err = math.sqrt(tag_omega_err_sq)
        except:
            log.warning(f'exception caught. tag_omega_err_sq={tag_omega_err_sq}')
            tag_omega_err = 0
        
    tag_omega_w_u = ufloat(tag_omega, tag_omega_err)
    
    tag_dil_w_u = 1. - 2*tag_omega_w_u

    tag_power = tag_eff*(1-2*tag_omega)**2
    try:
        tag_power_w_u = ufloat(tag_power, math.sqrt((tag_power*(1.-tag_power)/num_sig_cands)))
    except:
        log.warning(f'exception caught. tag_power={tag_power}, tag_eff={tag_eff}, tag_omega={tag_omega}')
        tag_power_w_u = ufloat(tag_power, 0)

    if print_result:
        print(f'Total Candidates:    {num_sig_cands}\n'+
              f'Selected Candidates: {N_tagged}\n'+
              f'Tagging Efficiency:  {tag_eff_w_u*100}%\n'+
              f'Mistag Probability:  {tag_omega_w_u*100}%\n'+
              f'Tagging Dilution:    {tag_dil_w_u*100}%\n'+
              f'Tagging Power:       {tag_power_w_u*100}%')

    return (tag_eff, tag_omega, tag_power)


class TagPowerCalc:
    """
    Class to handle the (average) flavour tagging estimation on a sample
    It requires the following inputs:

    @param df: input data frame
    
    @param query_str_wph: query to apply on the data frame. The tagging power
                          is computed 'after' applying this selection on
                          the data set. This query string has to satisfy
                          a syntax like the one of the following example:

                          query_str_wph =  'B_OSMuonDev_TagPartsFeature_P/1000 > {} & '+ \
                                           'B_OSMuonDev_TagPartsFeature_PT/1000 > {}'

                          i.e., the cut values have to be replaced by '{}'.
                          The actual cut values have to be passed via as
                          a numpy array via the 'evaluate' method

    @param num_evts: normalisation, i.e. number of events before any cut is applied

    @param pt_name: tagging particle pT column name in df

    Once an instance 'tagp' is created, the 'tagp.evaluate' function can be passed
    to a scipy or skopt optimiser/minimiser in order to get set on cuts giving the
    highest average tagging power.

    
    """
    
    def __init__(self, df, query_str_wph, num_evts, pt_name='B_OSMuonDev_TagPartsFeature_PT', weight_column='SigYield_sw'):
        self.df = df
        self.query_str_wph = query_str_wph
        self.num_evts = num_evts
        self.pt_name = pt_name
        self.weight_column = weight_column
        self.no_neg_cut_vals = True
        
    def evaluate(self, cut_vals, print_result=False, take_max_pt=False):
        if self.no_neg_cut_vals:
            if any(cut_val < 0. for cut_val in cut_vals):
                return 1e10 # limit to positive values to ensure pandas queries work
            #cut_vals_int = np.array(cut_vals).clip(min=0) # limit to positive values to ensure pandas queries work
            
        query_str = self.query_str_wph.format(*cut_vals)
        df_candidates = self.df.query(query_str)
        if not take_max_pt:
            df_candidates = calc_tagfracs_per_cand(df_candidates, self.weight_column)
        tagging_perf = calculate_tagging_performance(df_candidates,
                                                     self.num_evts,
                                                     self.weight_column,
                                                     self.pt_name,
                                                     print_result,
                                                     take_max_pt)
        return -tagging_perf[2]
