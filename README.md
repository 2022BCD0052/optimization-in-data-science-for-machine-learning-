# optimization-in-data-science-for-machine-learning
# Adam Optimization Algorithm

This project provides a simple implementation of the **Adam optimization algorithm** in Python. Adam (short for Adaptive Moment Estimation) is a popular algorithm used in machine learning to optimize models by iteratively adjusting parameters to minimize a loss function.

## Features

- **Adam Optimizer**: Implements the Adam algorithm for parameter optimization.
- **Customizable Parameters**:
  - **W_init**: Initial weight parameter.
  - **alpha**: Learning rate.
  - **beta1**: Smoothing parameter for the first moment.
  - **beta2**: Smoothing parameter for the second moment.
  - **epsilon**: Small constant for numerical stability.
  - **max_iters**: Maximum number of iterations.

- **Sample Loss Function**: Uses a simple smooth loss function \( L(W) = W^2 \) to demonstrate optimization.
- **Gradient Calculation**: Computes the gradient of the loss function for each iteration.

## Dependencies

- `numpy`: For numerical computations.
- `matplotlib`: For plotting the optimization progress.

## Usage

1. Clone or download this repository.
2. Ensure all dependencies are installed:
   ```bash
   pip install numpy matplotlib
   ```
3. Run the script:
   ```bash
   python adam.py
   ```
   
This will optimize the weight \(W\) to minimize the sample loss function and plot the progress.

## Parameters

You can customize the Adam optimizer parameters to experiment with different configurations:

W_init = 10.0  # Initial weight
alpha = 0.1    # Learning rate
beta1 = 0.9    # First moment decay rate
beta2 = 0.99   # Second moment decay rate
epsilon = 1e-8 # Small constant for numerical stability
max_iters = 1000  # Maximum iterations


## Example

By default, the script runs the optimizer on \(L(W) = W^2\) to demonstrate how the Adam algorithm minimizes the loss function over time.
