import matplotlib.pyplot as plt
import yaml
import numpy as np



with open("dicts.yml", mode="rt", encoding="utf-8") as file:
    dct = yaml.safe_load(file)[0]

# keys:    ['γ_protoc', 'maxiters', 'skip_factor', 'ω_protoc', 'nb_protocols', 'end_datetime', 'dct_NN', 'n_weights', 'nb_iter_optim', 'captureG', 'saveat', 'run', 'datetime', 'model_univ', 'Ncycles', 'start_datetime', 'T', 'nb_pts_per_cycle', 'γ', 'start_at', 'ω', 'tdnn_coefs', 'losses', 'dct_giesekus']

tdnn = dct["tdnn_coefs"]
#losses = dct["losses"]
#plt.plot(losses)
#plt.show()

tdnn = np.asarray(tdnn)
print(tdnn.shape)
print(tdnn)
