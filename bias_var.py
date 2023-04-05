from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# Generate a synthetic dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, noise=0.5, random_state=42)

# Create a decision tree regression model
tree = DecisionTreeRegressor(random_state=42)

# Create a bagging regressor with 100 trees
bagging = BaggingRegressor(tree, n_estimators=100, random_state=42)

# Evaluate the models with and without bagging
mse_without_bagging = cross_val_score(tree, X, y, scoring='neg_mean_squared_error', cv=5).mean()
mse_with_bagging = cross_val_score(bagging, X, y, scoring='neg_mean_squared_error', cv=5, n_jobs=-1).mean()

# Print the mean squared error for both cases
print(f"Mean squared error without bagging: {-mse_without_bagging:.2f}")
print(f"Mean squared error with bagging: {-mse_with_bagging:.2f}")


'''In this example, we first generate a synthetic dataset with 1000 samples and 10 features, where only 5 features are informative. We then create a random forest classification model with 100 trees using scikit-learn's RandomForestClassifier class. We evaluate the model using 5-fold cross-validation, both with and without bagging (by setting n_jobs=-1 to use all available CPU cores).

Finally, we print the accuracy for both cases. Since bagging reduces variance, we would expect the accuracy with bagging to be higher than the accuracy without bagging.'''


from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)

# Create a random forest classification model with 100 trees
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Create a bagging classifier with the random forest model
bagging = BaggingClassifier(rf, n_estimators=100, random_state=42)

# Evaluate the models with and without bagging
log_loss_without_bagging = cross_val_score(rf, X, y, scoring='neg_log_loss', cv=5, n_jobs=-1).mean() * -1
log_loss_with_bagging = cross_val_score(bagging, X, y, scoring='neg_log_loss', cv=5, n_jobs=-1).mean() * -1

# Calculate the error reduction percentage with bagging
error_reduction = (log_loss_without_bagging - log_loss_with_bagging) / log_loss_without_bagging * 100

# Print the results
print(f"Log loss without bagging: {log_loss_without_bagging:.2f}")
print(f"Log loss with bagging: {log_loss_with_bagging:.2f}")
print(f"Error reduction with bagging: {error_reduction:.2f}%")


import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, Dropdown, IntSlider
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, BaggingRegressor
from sklearn.model_selection import train_test_split

# Define the interactive function
@interact(problem=Dropdown(options=['Classification', 'Regression'], value='Classification', description='Problem'),
          n_estimators=IntSlider(min=1, max=200, step=1, value=10, description='Number of Learners'))
def plot_decision_boundary(problem, n_estimators):
    # Generate synthetic data
    if problem == 'Classification':
        X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, random_state=42)
        model = BaggingClassifier(RandomForestClassifier(n_estimators=1, max_depth=2), n_estimators=n_estimators, random_state=42)
    elif problem == 'Regression':
        X, y = make_regression(n_samples=500, n_features=1, noise=10, random_state=42)
        model = BaggingRegressor(RandomForestRegressor(n_estimators=1, max_depth=2), n_estimators=n_estimators, random_state=42)
    
    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', alpha=0.5)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    if problem == 'Classification':
        plt.title(f'Decision boundary for BaggingClassifier with {n_estimators} learners')
    elif problem == 'Regression':
        plt.title(f'Decision boundary for BaggingRegressor with {n_estimators} learners')
    plt.show()

