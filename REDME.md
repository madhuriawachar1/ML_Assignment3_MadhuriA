# Introduction
 Network URL: http://10.7.43.141:8501

 
Bagging can reduce variance in model because of bootstraping and aggregation without impacting the bias.
in bagging each model is low bias and high variance model.


Bagging is a technique that involves training multiple models on different subsets of the training data and then aggregating their predictions to obtain a final prediction. 

By using thesemultiple models trained, bagging reduces the variance of the overall model, which can improve its generalization performance on new, unseen data.


* create model

In the above code, ```the create_model function ```creates a bagging model for either classification or regression.

 The function takes an argument ```n_estimators```, which determines the number of learners to use in the bagging model. The more learners there are in the bagging model, the lower the variance of the overall model is likely to be.

To demonstrate how bagging reduces variance, the Streamlit app allows the user to adjust the number of learners using a slider in the sidebar menu. As the number of learners increases, the variance of the model is expected to decrease, as the aggregate prediction becomes more robust to the idiosyncrasies of any individual model.

In the app, the decision boundary visualizations show how the bagging model changes as the number of learners is increased. 

### For classification
#### dataset
X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                                    n_redundant=0, n_clusters_per_class=1, random_state=42)
    


the decision boundary becomes smoother and less sensitive to small changes in the input features as the number of learners increases.

### For regression

#### dataset
 X, y = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)


 the prediction becomes smoother and less erratic as the number of learners increases, as the aggregate prediction becomes less sensitive to the specific training data used to fit each individual model.

Overall, the app demonstrates how bagging can reduce variance for both classification and regression, and how this is demonstrated by the number of learners used in the bagging model.










