"""
Run the model with the original moments
"""
import numpy as np
import pandas as pd
import pybobyqa

from norpy import simulate
from norpy.model_spec import get_random_model_specification, get_model_obj
from smm_prep import get_moments, get_weigthing_matrix
from norpy.adapter.SimulationBasedEstimation import SimulationBasedEstimationCls
from optimizers.auxiliray_pyogba import wrapper_pybobyqa
from auxiliary import moments_final, weigthing_final

# Container for optimization
optim_paras = {
    "coeffs_common": slice(0,2),
    "coeffs_home": slice(2,4),
    "coeffs_edu": slice(4, 10),
    "coeffs_work": slice(10, 22),
    "type_spec_shifts": slice(22, 31),
    "shocks_cov": slice(31, 40),

}
pos_dict = {
    "coeffs_common": None,
    "coeffs_home": [0, 2],
    "coeffs_edu": list(range(6)),
    "coeffs_work": list(range(0, 7)) + list(range(8, 13)),
    "type_spec_shifts": list(range(3, 12)),
    "shocks_cov": None
}

box_dict = {
    "coeffs_common": [(-15000, 5000),
                      (-15000, 50000)],
    "coeffs_home": [(0.0, 250000.0),
                    (1000.0, 6000.0)
                    ],
    "coeffs_edu": [(-16000.0, -5000.0),
                   (-1000000.0, 500000.0),
                   (-1500000.0, 1000),
                   (-300000.0, -100000.0),
                   (-400000.0, 20000.0),
                   (-300000.0, -1000.0)


                   ],
    "coeffs_work": [(10.5, 12.1),
                    (0.0, 0.2),
                    (0.0, 0.05),
                    (-0.04, 0.0),
                    (0.0, 0.15),
                    (0.0, 0.1),
                    (-0.1, 0.1),
                    (0.0, 0.2),
                    (0.0, 0.5),
                    (-100000.0, 100000.0),
                    (-250000.0, 100000.0),
                    (-100000.0, 100000.0),
                    ],
    "type_spec_shifts": [(-1, 1.0),
                         (-100000.0, 100000.0),
                         (-100000.0, 100000.0),
                         (-1, 1.0),
                         (-100000.0, 100000.0),
                         (-100000.0, 100000.0),
                         (-1, 1.0),
                         (-100000.0, 100000.0),
                         (-100000.0, 100000.0),
                         ],
    "shocks_cov": [(-10, 10),
                   (-100000, 100000),
                   (-100000, 100000),
                   (-100000, 100000),
                   (-1000000000, 100000000000),
                   (-100000, 100000),
                   (-100000, 100000),
                   (-100000, 100000),
                    (-10000000000, 100000000000)
                   ]

}

initialization_object = get_random_model_specification(
    constr={"num_periods": 49,
            "num_agents_sim": 5000,
            "num_draws_emax": 5000,
            "num_types": 4,
            "num_edu_start": 1,
            "edu_max": 25,
            "seed_sim": 132,
            "seed_emax": 456,
            "edu_range_start": np.array([9]),
            "type_prob_cond_schooling": np.array([1 / 4] * 4).reshape(1, 4),
            "intial_lagged_schooling_prob": float(1),
            "delta": 0.926,
            #"shocks_cov": np.array([1.90836000e-01, -4.06073119e+04, -4.33622160e+04,
            #                        -4.06073119e+04, 5.70574437e+10, -5.68684800e+04,
            #                        -4.33622160e+04, -5.68684800e+04, 3.15822658e+10]),
            #"shocks_cov":np.array([1.02400000e-01, 7.64376499e+04, 5.68702352e+04,
            #                       7.64376499e+04, 5.70577570e+10, 4.24514368e+10,
            #                       5.68702352e+04, 4.24514368e+10, 3.15842153e+10]),
            "shocks_cov":np.array([ 1.02400000e-01, -5.47200000e-02, -7.80800000e-02,
                                   -5.47200000e-02,  5.70577589e+10, -7.64376094e+04,
                                   -7.80800000e-02, -7.64376094e+04,  3.15842153e+10]),
            "coeffs_home": np.array([118298.619219974672887,
                                     0,
                                     5002.729695187197649
                                     ]),
            "coeffs_edu": np.array([-12095.401873979237280,
                                    -299318.752652849536389,
                                    -145180.005109090765473,
                                    -264622.915707175270654,
                                    -230544.888455883046845,
                                    -7246.006801301322412,
                                    0
                                    ]),

            "coeffs_work": np.array([11.208202053369439,
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

            "coeffs_common": np.array([-12008.359773063100874,
                                       48942.700331236468628
                                       ]),
            "type_spec_shifts": np.array([0,
                                          0,
                                          0,
                                          0.176162183496461,
                                          9662.726586080767447,
                                          -5137.074427099956665,
                                          -0.520489768432775,
                                          -46004.947807452808775,
                                          11954.110836926454795,
                                          0.066716619965600,
                                          -31343.568694616944413,
                                          -7246.492439852168900
                                          ])

            }
)
# define paras for optimizer
max_evals = 1000000

# Now we start with the optimization
args = (
    initialization_object,
    moments_final,
    weigthing_final,
    get_moments,
    optim_paras,
    pos_dict,
    max_evals
)
adapter_smm = SimulationBasedEstimationCls(*args)


box_lower = np.array([y[0] for x in box_dict.keys() for y in box_dict[x] ])
box_upper = np.array([y[1] for x in box_dict.keys()for y in box_dict[x] ])

kwargs = dict()
kwargs['scaling_within_bounds'] = True
kwargs['bounds'] = (box_lower, box_upper)
kwargs['objfun_has_noise'] = True
# kwargs['maxfun'] = 100
kwargs['maxfun'] = 10e6

a = simulate(get_model_obj(initialization_object)).replace({"wages":{-99:np.nan}})


rslt = pybobyqa.solve(adapter_smm.evaluate, adapter_smm.free_params, **kwargs)
