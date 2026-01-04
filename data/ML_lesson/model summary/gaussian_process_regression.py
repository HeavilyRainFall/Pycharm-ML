import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(42)
np.random.seed(42)

# Define true function (sine function)
def true_function(x):
    return torch.sin(2 * np.pi * x)

# Generate virtual training data: sample uniformly in [0, 1] interval with added noise
n_train = 200
train_x = torch.linspace(0, 1, n_train)
noise_std = 0.2
train_y = true_function(train_x) + noise_std * torch.randn(train_x.size())

# Generate test data: for prediction and plotting
n_test = 500
test_x = torch.linspace(0, 1, n_test)

# Define Gaussian Process model (using gpytorch's ExactGP)
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        # Mean function: using constant mean
        self.mean_module = gpytorch.means.ConstantMean()
        # Covariance function: using RBF kernel wrapped with ScaleKernel
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

# Set model to training mode
model.train()
likelihood.train()

# Define optimizer (Adam) and marginal log likelihood loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Record training loss
training_loss = []
n_iter = 100  # Training iterations

# Training process
for i in range(n_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)  # We minimize negative log marginal likelihood
    loss.backward()
    optimizer.step()
    training_loss.append(loss.item())
    if (i+1) % 10 == 0:
        print(f'Iteration {i+1}/{n_iter} - Loss: {loss.item():.3f}')

# Training complete, switch to evaluation mode
model.eval()
likelihood.eval()

# Prediction: compute posterior distribution on test data
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_pred = likelihood(model(test_x))
    pred_mean = test_pred.mean
    pred_var = test_pred.variance
    lower, upper = test_pred.confidence_region()

# Plot graphs: 4 subplots in one figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Gaussian Process Regression Example', fontsize=16)

# Subplot 1: Training data and true function
axes[0, 0].scatter(train_x.numpy(), train_y.numpy(), color='red', s=50, label='Training Data')
axes[0, 0].plot(test_x.numpy(), true_function(test_x).numpy(), color='blue', linewidth=2, label='True Function')
axes[0, 0].set_title('Training Data and True Function', fontsize=12)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
axes[0, 0].legend()
axes[0, 0].grid(True, linestyle='--', alpha=0.6)

# Subplot 2: GP predicted mean and confidence interval
axes[0, 1].plot(test_x.numpy(), pred_mean.numpy(), color='magenta', linewidth=2, label='Predicted Mean')
axes[0, 1].fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), color='magenta', alpha=0.3, label='95% Confidence Interval')
axes[0, 1].scatter(train_x.numpy(), train_y.numpy(), color='red', s=40, label='Training Data', zorder=10)
axes[0, 1].set_title('GP Predicted Mean and Confidence Interval', fontsize=12)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('y')
axes[0, 1].legend()
axes[0, 1].grid(True, linestyle='--', alpha=0.6)

# Subplot 3: Predicted variance (uncertainty)
axes[1, 0].plot(test_x.numpy(), pred_var.numpy(), color='green', linewidth=2, label='Predicted Variance')
axes[1, 0].set_title('Predicted Variance (Uncertainty)', fontsize=12)
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('Variance')
axes[1, 0].legend()
axes[1, 0].grid(True, linestyle='--', alpha=0.6)

# Subplot 4: Training loss curve of negative log marginal likelihood
axes[1, 1].plot(range(1, n_iter+1), training_loss, color='orange', linewidth=2, label='Training Loss')
axes[1, 1].set_title('Negative Log Marginal Likelihood Loss Curve', fontsize=12)
axes[1, 1].set_xlabel('Iteration')
axes[1, 1].set_ylabel('Negative Log Marginal Likelihood')
axes[1, 1].legend()
axes[1, 1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("Gaussian Process Regression Model ran successfully!")
print(f"Training completed, final loss value: {training_loss[-1]:.3f}")
print(f"Predicted mean range: [{pred_mean.min():.3f}, {pred_mean.max():.3f}]")
print(f"Predicted variance range: [{pred_var.min():.3f}, {pred_var.max():.3f}]")