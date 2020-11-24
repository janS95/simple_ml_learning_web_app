import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import matplotlib.pyplot as plt



def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K",1,15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C",0.01,10.0)
        params["C"] = C
    elif clf_name == "Decision Tree":
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        params["max_depth"] = max_depth
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1,100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    elif clf_name == "Decision Tree":
        clf = DecisionTreeClassifier(max_depth=params["max_depth"],random_state=1)
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                        max_depth=params["max_depth"],random_state=1)
    return clf

def scale_data(X,method):
    if method == "Standard Scaler":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif method == "Min Max Scaler":
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    else:
        X = X
    return X

st.title("Simple Machine Learning Web App")

st.write("""
## Explore different Machine Learning classifier
""")
st.write("""
        """)

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris","Breast Cancer","Wine"))

classifier_name = st.sidebar.selectbox("Select Classifier",
                                    ("KNN","SVM","Decision Tree","Random Forest"))

scaling_method = st.sidebar.selectbox("Select Feature-Scaling Method",
                                    ("None","Standard Scaler","Min Max Scaler"))

X, y = get_dataset(dataset_name)
st.write("shape of dataset", X.shape)
st.write("number of classes", len(np.unique(y)))

X = scale_data(X,scaling_method)

params = add_parameter_ui(classifier_name)



clf = get_classifier(classifier_name,params)

# Classification
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)

clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)

acc = accuracy_score(y_test,y_predict)
st.write(f"classifier = {classifier_name}")
st.write(f"accuracy = {acc}")

# PLOT
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)

# TODO
# Add more parameters
# Add more classifier
# Add visualization of classifier boundaries


