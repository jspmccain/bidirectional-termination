import sys

from sim_functions import *

filename = sys.argv[1]

## make a gene index map
gene_map_test = make_gene_body_index(gene_body_length=1000, terminator_length=3)

## set parameters
par_lookup_out_strong_term = make_look_up_for_gene_body(stay_prob_gb = 0.00999, 
                                            forward_prob_gb = 0.99, 
                                            release_prob_gb = 0.00001,
                                            stay_prob_term = 0.2, 
                                            forward_prob_term = 0.8, 
                                            release_prob_term = 0,
                                            stay_prob_ogb = 0,
                                            forward_prob_ogb = 0.8,
                                            release_prob_ogb = 0.2)
par_lookup_out_weak_term = make_look_up_for_gene_body(stay_prob_gb = 0.00999, 
                                            forward_prob_gb = 0.99, 
                                            release_prob_gb = 0.00001,
                                            stay_prob_term = 0.8, 
                                            forward_prob_term = 0.2, 
                                            release_prob_term = 0,
                                            stay_prob_ogb = 0,
                                            forward_prob_ogb = 0.8,
                                            release_prob_ogb = 0.2)

## run forward model with strong and weak terminator
parameter_scan_out_strong = parameter_runs(gene_map = gene_map_test, par_list = par_lookup_out_strong_term, 
                                       on_rate_per_time_step_rev_list = [100, 500, 1000, 2000, 5000], 
                                       on_rate_per_time_step_for_list = [100, 500, 1000, 2000, 5000],
                                       n_steps = 50000, bump_distance = 20, unidirectional=False)
parameter_scan_out_weak = parameter_runs(gene_map = gene_map_test, par_list = par_lookup_out_weak_term, 
                                       on_rate_per_time_step_rev_list = [100, 500, 1000, 2000, 5000], 
                                       on_rate_per_time_step_for_list = [100, 500, 1000, 2000, 5000],
                                       n_steps = 50000, bump_distance = 20, unidirectional=False)

filename_strong = '../data/parameter_scan_out_strong20_noncanonterm1' + filename + '.csv'
filename_weak = '../data/parameter_scan_out_weak20_noncanonterm2' + filename + '.csv'

## save the runs
parameter_scan_out_strong.to_csv(filename_strong)
parameter_scan_out_weak.to_csv(filename_weak)
