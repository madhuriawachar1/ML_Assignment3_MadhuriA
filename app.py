'''import streamlit as st



st.set_page_config(page_title="Ensemble",layout="wide")



st.subheaderst.title("HREJHBDJKNV DSHBDIUFHI")

# Define a dropdown menu to select classification or regression
option = st.sidebar.selectbox('Select a task:', ('Classification', 'Regression'))

# Define a slider to adjust the number of learners
n_learners = st.sidebar.slider('Number of learners:', 1, 100, 10)

if option == 'Classification':
    model = BaggingClassifier(base_estimator=base_estimator, n_estimators=n_learners)
else:
    model = BaggingRegressor(base_estimator=base_estimator, n_estimators=n_learners)

model.fit(X_train, y_train)

# Display the selected options
st.write('Selected task:', option)
st.write('Number of learners:', n_learners)'''

