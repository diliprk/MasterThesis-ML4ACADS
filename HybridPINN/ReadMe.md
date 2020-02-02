
The two basic approaches for the Hybrid PINN architecture are contained in the following `.py` scripts:
* `Hybrid_PINN_v1_x.py` - Refers to the **GeneralApproach** where we use normal feature scaling, `ReLu` activations in the hidden layer and random weight initializations and gradient updates with fixed learning rate.
* `Hybrid_PINN_v2_x.py`- Refers to the **FE_LS** approach with customized Feature Engineering (`FE`), _linear activation_ in all layers,  lesser no. of neurons in the input and hidden layers, preset weight initialization and Line Search (`LS`) method to find the adaptive learning rate. This method gives more precise and faster convergence of the flowrates in lesser training iterations, albeit at a slight expense of more FDDN calls. 

The jupyter notebooks (`.ipynb` files) in this folder imports one of the above python scripts depending on the approach (**General** or **FE_LS**). Each notebook has a customized training function call to predict one or two correction factors. The sub-variants in the filenames of the `.ipynb` files, like:

 - **SingleHoVLoop** - refers to the training of the neural network one point at a time sequentially (i.e. [online learning](https://en.wikipedia.org/wiki/Online_machine_learning)), for optimal convergence, before moving to the next point. This yields optimal `c_fs` in a short span of time.
 - **miniBatch** - refers to the training of the neural network using batch learning or minibatch gradient descent, where we use  `batch_size >= 2`. miniBatch training is necessary so that we don't end up with an overfit model as we would get in the online (SingleHoVLoop) learning method. The generalized model which we will get in the miniBatch training is expected to give better predictions for existing and new points.

The `FDDN_Lib` script runs the FDDN Solver in batch mode (`run_fddn.bat`) and save the data frame results in a .csv file.

#### Misc Implementations:

 1. An ***Early stopping*** feature is implemented in the **GeneralApproach** which monitors the change in the `error_loss` function for the last 10 epochs and terminates the training if there is no improvement. This can be seen in action in the `PINN_v1.0_GeneralApproach_SingleHoVLoop_OneCF.ipynb` file.
 2.  All approaches also use an ***Adaptive Batch Size*** technique during the training process that removes the converged points from the training data population and also lowers the batch size when the available *training data points is < initial batch size*

