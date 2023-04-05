import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, Dropdown, IntSlider
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, log_loss

# Define the interactive function
@interact(problem=Dropdown(options=['Classification', 'Regression'], value='Classification', description='Problem'),
          n_estimators=IntSlider(min=1, max=200, step=1, value=10, description='Number of Learners'))
def plot_decision_boundary(problem, n_estimators):
    # Generate synthetic data
    if problem == 'Classification':
        X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, random_state=42)
        model = BaggingClassifier(RandomForestClassifier(n_estimators=1, max_depth=2), n_estimators=n_estimators, random_state=42)
        error = log_loss
    elif problem == 'Regression':
        X, y = make_regression(n_samples=500, n_features=1, noise=10, random_state=42)
        model = BaggingRegressor(RandomForestRegressor(n_estimators=1, max_depth=2), n_estimators=n_estimators, random_state=42)
        error = mean_squared_error
    
    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # Compute the error
    y_pred = model.predict(X_test)
    err_without_bagging = error(y_test, y_pred)
    
    y_pred_bagging = np.zeros_like(y_test, dtype=np.float64)
    for estimator in model.estimators_:
        y_pred_bagging += estimator.predict(X_test)
    y_pred_bagging /= n_estimators
    err_with_bagging = error(y_test, y_pred_bagging)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 6))
    if problem == 'Classification':
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', alpha=0.5)
        
        
    elif problem == 'Regression':
        plt.scatter(X, y, color='blue', alpha=0.5)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx = np.linspace(x_min, x_max, 500).reshape(-1, 1)
    
    
    if problem == 'Classification':
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        yy = np.linspace(y_min, y_max, 500).reshape(-1, 1)
        xx, yy = np.meshgrid(xx, yy)
        X_mesh = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict_proba(X_mesh)[:, 1].reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
    elif problem == 'Regression':
        yy = model.predict(xx)
        plt.plot(xx, yy, color='red', alpha=0.8)
    plt.xlabel('Feature')
    
    # Print the results
    if problem == 'Classification':
        print(f'Log loss without bagging: {err_without_bagging:.2f}')
        print(f'Log loss with bagging: {err_with_bagging:.2f}')
        
        
plot_decision_boundary('Classification', 200)
