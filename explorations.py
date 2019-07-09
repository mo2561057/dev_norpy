import numpy as np
import pandas as pd
import pybobyqa

from norpy import simulate
from norpy.model_spec import get_random_model_specification, get_model_obj
from norpy.simulate.simulate import simulate
from smm_prep import get_moments, get_weigthing_matrix
from norpy.adapter.SimulationBasedEstimation import SimulationBasedEstimationCls
from optimizers.auxiliray_pyogba import wrapper_pybobyqa
from auxiliary import weigthing_final, moments_final

# Define which paramteres to optimize over!
# Which one do we start with anyways
# Shall we specify a full contraint vector ????????
optim_paras = {
    "coeffs_common": slice(0, 2),
    "coeffs_home": slice(2, 5),
    "coeffs_edu": slice(5, 12),
    "coeffs_work": slice(12, 25),
    "type_spec_shifts": slice(25, 37),
    "shocks_cov":slice(37, 46),
#    "delta":slice(34, 35)

}

initialization_object = get_random_model_specification(
    constr = {"num_periods":49,
              "num_agents_sim":5000,
              "num_emax":5000,
              "num_types":4,
              "num_edu_start":1,
              "edu_max":25,
              "seed_sim":132,
              "seed_emax":456,
              "edu_range_start":np.array([9]),
              "type_prob_cond_schooling":np.array([1/4]*4).reshape(1,4),
              "intial_lagged_schooling_prob": float(1),
              "delta":0.926,
              "shocks_cov":np.array([1.90836000e-01, -4.06073119e+04, -4.33622160e+04,
                                     -4.06073119e+04,  5.70574437e+10, -5.68684800e+04,
                                     -4.33622160e+04, -5.68684800e+04,  3.15822658e+10]),
              "coeffs_home": np.array([118298.619219974672887,
                                       0,
                                       5002.729695187197649
                                       ]),
              "coeffs_edu":np.array([-12095.401873979237280,
                                     -299318.752652849536389,
                                     -145180.005109090765473,
                                     -264622.915707175270654,
                                     -230544.888455883046845,
                                     -7246.006801301322412,
                                     0
                                     ]),
              "coeffs_work":np.array([11.208202053369439,
                                      0.097513358678320,
                                      0.018614181282294,
                                      -0.029874125213061,
                                      0.039423863168125,
                                      0.030096500914825,
                                      0.000604371633589,
                                      0.000000000000000,
                                      0.102604480018636,
                                      0.316817020463839,
                                      35975.477362620003987,
                                      -174807.501464794069761,
                                      34831.665739499643678]),
              "coeffs_common":np.array([-12008.359773063100874 ,
                                        48942.700331236468628
                                       ]),
              "type_spec_shifts": np.array([0,
                                            0,
                                            0,
                                            0.176162183496461,
                                            9662.726586080767447,
                                            -5137.074427099956665,
                                            -0.520489768432775 ,
                                            -46004.947807452808775,
                                            11954.110836926454795,
                                            0.066716619965600 ,
                                            -31343.568694616944413,
                                            -7246.492439852168900
                                            ])

              }
   )

#Simulate the respective dataset!
sim_df = simulate(get_model_obj(initialization_object))
#Quick fix for now
sim_df["wages"] = sim_df["wages"].replace({-99:np.nan})
sim_df["identifier"] = sim_df["agent"]

#get observed moments
moment_obs = get_moments(sim_df)

#specs for weighting matrix#
#Be sure to set a number smaller than that
num_agents_smm = initialization_object["num_agents_sim"]/4
num_boots = 50

#Create the weighting_matrix
#weighting_matrix = get_weigthing_matrix(sim_df, num_boots, num_agents_obs)

#Set max evals
max_evals = 1000


# Now we start with the optimization
args = (
    initialization_object,
    moments_final,
    weigthing_final,
    get_moments,
    optim_paras,
    max_evals,
)
adapter_smm = SimulationBasedEstimationCls(*args)


#Get bounds for estimation
box_constraints = [
    (-5000,5000),
    (-5000,5000),
    (0.0,250000.0),
    (0,0.00001),
    (1000.0,6000.0),
    (-100000.0,500000.0),
    (-1500000.0,1000),
    (-300000.0,-100000.0),
    (-400000.0,20000.0),
    (-300000.0,-1000),
    (-100000.0,-1000),
    (10.5,12.1),
    (0.0, 0.2),
    (0.0, 0.05),
    (-0.03,0.0),
    (0.0,0.15),
    (0.0,0.1),
    (0.0,0.1),
    (0,0.0000001),
    (0.0, 0.2),
    (0.0,0.5),
    (-100000.0,100000.0),
    (-200000.0,100000.0),
    (-100000.0, 100000.0),
    (0.0001),
    (0,0.00001),
    (0.00001),
    (-0.5,1.0),
    (-100000.0,100000.0),
    (-100000.0,100000.0),
    (-0.5,1.0),
    (-100000.0,100000.0),
    (-100000.0,100000.0),
    (-0.5,1.0),
    (-100000.0,100000.0),
    (-100000.0,100000.0),
    (-1,1),
    (-10000000000,1000000000),
    (-1000000000, 1000000000),
    (-1000000000, 1000000000),
    (-1000000000, 1000000000),
    (-1000000000, 100000000),
    (-1000000000, 100000000),
    (-1000000000, 100000000),
    (-10000000000, 100000000)
    ]

kwargs = dict()
kwargs['scaling_within_bounds'] = True
kwargs['bounds'] = box_constraints
kwargs['objfun_has_noise'] = True
kwargs['maxfun'] = 10e6


#Estimate
#rslt = pybobyqa.solve(adapter_smm.evaluate, adapter_smm.free_params)
