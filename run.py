"""
Run the model with the original moments
"""
import numpy as np
import pandas as pd
import pybobyqa

from norpy import simulate
from norpy.model_spec import get_random_model_specification
from smm_prep import get_moments, get_weigthing_matrix
from norpy.adapter.SimulationBasedEstimation import SimulationBasedEstimationCls
from optimizers.auxiliray_pyogba import wrapper_pybobyqa
from auxiliary import moments_final, weigthing_final

#Container for optimization
optim_paras = {
    "coeffs_common": slice(0, 2),
    "coeffs_home": slice(2, 5),
    "coeffs_edu": slice(5, 12),
    "coeffs_work": slice(12, 25),
    "type_spec_shifts": slice(25, 34),
    "shocks_cov":slice(25, 34),
#    "delta":slice(34, 35)

}

initialization_object = get_random_model_specification(constr = {"num_periods":49,
                                                                 "num_agents_sim":5000,
                                                                 "num_types":3,
                                                                 "num_edu_start":1,
                                                                 "edu_range_start":np.array([9]),
                                                                 "type_prob_cond_schooling":np.array([1/3]*3).reshape(1,3),
                                                                 "initial_lagged_schooling_prob":0.5,
                                                                 "delta":0.96
                                                                 },
                                                       p_constr = {"coeffs_edu": [6,0]}
                                                       )
#define paras for optimizer
max_evals = 100000

# Now we start with the optimization
args = (
    initialization_object,
    moments_final,
    weigthing_final,
    get_moments,
    optim_paras,
    max_evals
)
adapter_smm = SimulationBasedEstimationCls(*args)




rslt = pybobyqa.solve(adapter_smm.evaluate, adapter_smm.free_params)