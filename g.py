import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from mlxtend.plotting import plot_decision_regions

# Define a function to create the data
def create_data(option):
    if option == 'Classification':
        X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                                    n_redundant=0, n_clusters_per_class=1, random_state=42)
    else:
        X, y = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)
    return X, y

# Define a function to create the bagging model
def create_model(option, n_learners):
    if option == 'Classification':
        base_estimator = DecisionTreeClassifier()
        model = BaggingClassifier(base_estimator=base_estimator, n_estimators=n_learners)
    else:
        base_estimator = DecisionTreeRegressor()
        model = BaggingRegressor(base_estimator=base_estimator, n_estimators=n_learners)
    return model

# Define the Streamlit app
st.title('Bagging Demo')

# Define a dropdown menu to select classification or regression
option = st.sidebar.selectbox('Select a task:', ('Classification', 'Regression'))

# Define a slider to adjust the number of learners
n_learners = st.sidebar.slider('Number of learners:', 1, 100, 10)

# Create the data and the bagging model
X, y = create_data(option)
model = create_model(option, n_learners)

# Fit the model to the data
model.fit(X, y)

# Compute the decision boundary for the bagging model
if option == 'Classification':
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
else:
    xx = np.arange(X.min(), X.max(), 0.1).reshape(-1, 1)
    Z = model.predict(xx)

# Visualize the data and the decision boundary
fig, ax = plt.subplots()
if option == 'Classification':
    plot_decision_regions(X, y, model, ax=ax)
else:
    ax.plot(X, y, 'o', label='data')
    ax.plot(xx, Z, label='bagging')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.legend()
st.pyplot(fig)
