This repository contains the code to simulate toy jets and run an RNN to fit the jet vertex positions.

The toy jets can be created using `python run_toy_simulation.py`

Once this is done, the RNN can be trained with `python RNNJF.py`. See the script for the various options available.

The file notebooks/Analysis.ipynb is used to evaluate the performance and create various plots and figures. The file notebooks/an_rnn_example.ipynb is a simple example of pre-processing the toy jet data, training an RNN, and then evaluating performance.