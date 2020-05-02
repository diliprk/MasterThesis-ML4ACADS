The code for running the Hybrid PINN implementation is segregated into 3 types of files:<br>
i.) **FDDN Library scripts** - `TZ6_FDDN_Lib.py` (or) `TZ3_FDDN_Lib.py`
These python scripts contain the FDDN related functions that reads and modifies the `.fddn file` (in json format) and returns the flow rates output from the FDDN Solver as a dataframe.<br>

ii.) **PINN Model configuration python scripts**
Each PINN model configuration is stored in a separate python script which is typically named after the model configuration (Eg: `TZ3_8CF.py`). This python script contains the class PINN which houses all the necessary functions to run the Hybrid PINN, such as:
● NormbyMax to scale input features according to the max value
● Initialize weights, biases and gradients of the model
● Adaptive Line Search related functions to find $\alpha_2$
● Forward propagation and backward propagation functions
● FDDN Solver Processes
● Update model parameters with the gradients from the backward pass
● Make new predictions for validation and test points.

The above core functions are wrapped in the `train_1p_seq` (for _single data point training_) or `train_batch` (for _full batch training_) which can be called to start the whole training process.<br>

iii.) **Jupyter notebooks (.ipynb files)**
The jupyter notebook provides the front end interface for the user to execute the functions in the above python scripts and view the results. In the jupyter notebooks we load datasets, compute loss coefficients from CFD estimates, train PINN network, visualize train and validation loss performance and finally make predictions for test data points (if trained in _full batch_ mode)
The filenames of these .ipynb files can end with the suffix _SingleDataPointTraining_ or _fullBatch_ denoting if the Hybrid PINN architecture was called to run in single data point training or full batch training modes.<br>

**Notes**:<br>
1.) The Hybrid PINN requires the `FDDN Solver - version 7.0.0.107` which can be installed after approval from Airbus.<br>
2.) Please install all the required python libraries - `pandas, numpy, SciPy, scikit-learn, bokeh, plotly, selenium` to run this Hybrid Architecture<br>
3.) Please make sure to edit the location in the `TZ6_run_fddn.bat` and `TZ3_run_fddn.bat` files to your current working directory
