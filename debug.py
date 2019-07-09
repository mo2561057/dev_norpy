"""
Script to debug the Inadmissibility

"""
import numpy as np
import pandas as pd
import pybobyqa
from collections import OrderedDict

from norpy import simulate
from norpy.model_spec import get_random_model_specification, get_model_obj
from smm_prep import get_moments, get_weigthing_matrix
from norpy.adapter.SimulationBasedEstimation import SimulationBasedEstimationCls
from optimizers.auxiliray_pyogba import wrapper_pybobyqa
from auxiliary import moments_final, weigthing_final

a = OrderedDict([('num_periods', 49),
                     ('num_types', 4),
                     ('num_draws_emax', 14),
                     ('num_edu_start', 1),
                     ('edu_range_start', np.array([9])),
                     ('edu_max', 25),
                     ('type_prob_cond_schooling', np.array([[0.25, 0.25, 0.25, 0.25]])),
                     ('shocks_cov', np.array([[3.75662695e+08, 4.60489420e+08, -6.21734211e+08],
                                              [-2.83447372e+08, 5.74118020e+10, -3.95221691e+07],
                                              [1.16576632e+09, -1.18342243e+08, 3.15930171e+10]])),
                     ('type_spec_shifts', np.array([[2.17524306e+08, -2.12531959e+08, 6.90996458e+08],
                                                    [-4.25670398e+08, -2.17177634e+07, 3.30555287e+08],
                                                    [-2.67687142e+08, -3.90385626e+08, 3.99015399e+08],
                                                    [1.53250927e+09, 1.38052338e+09, -5.38477985e+08]])),
                     ('coeffs_common', np.array([-1.15478159e+09, -6.77716360e+08])),
                     ('coeffs_work', np.array([-1.55541425e+07, -1.75999819e+08, 1.72783318e+09, 1.55680193e+08,
                                               1.56680600e+08, -1.12457330e+09, -6.35814587e+08, 8.61671654e+08,
                                               -1.18778995e+09, 1.03322335e+09, 6.95373650e+08, 1.26242114e+08,
                                               1.95130469e+09])),
                     ('coeffs_home', np.array([1.65508699e+09, -2.02334835e+09, -1.22627928e+09])),
                     ('coeffs_edu', np.array([-4.28148569e+08, -8.56720420e+08, 9.30894513e+07, 5.40938035e+08,
                                              3.83618266e+08, -3.88566534e+08, 7.33655781e+08])),
                     ('num_agents_sim', 5000), ('delta', 0.926), ('seed_sim', 132), ('seed_emax', 456),
                     ('intial_lagged_schooling_prob', 1.0)])

b = get_model_obj(a)

b = simulate(b)
