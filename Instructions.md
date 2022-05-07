The codes of Liearized Learning. For different cases, the excutable codes are in the corresponding folder. In this zip file, theres is only one case given. The excutable codes are in `msnn_mix_40_35` and `mix` means the mixed scheme is used in the paper.

To start training run
`python StokesEquationSolver.py --epochs=1000`

The data generation process is concluded in the `utils.utils.data_generator`, which will be run for every training epoch.

In the `NS_msnn.py`, there are several functions & class used during the whole training process, Like `train` function and the neural network class `MultiScaleNet`.

In the `plots.py`, you could plot the outputs for specific epoch.
