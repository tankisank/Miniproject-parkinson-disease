from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report,roc_auc_score,confusion_matrix
def load_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
    data = pd.read_csv(url)
    data.drop('name', axis=1, inplace=True)
    X = data.drop('status', axis=1)
    y = data['status']
    return X, y

def knn_model(X, y):
    knn = KNeighborsClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    svm_roc=roc_auc_score(y_test, y_pred)
    svm_recall=recall_score(y_test,y_pred)
    svm_precision=precision_score(y_test,y_pred)
    return accuracy_score(y_test,y_pred),svm_roc,svm_recall,svm_precision


def lr_model(X, y):
    lr = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    lr.fit(X_train, y_train)
    y_pred=lr.predict(X_test)
    svm_roc=roc_auc_score(y_test, y_pred)
    svm_recall=recall_score(y_test,y_pred)
    svm_precision=precision_score(y_test,y_pred)
    return accuracy_score(y_test,y_pred),svm_roc,svm_recall,svm_precision

def svm_model(X, y):
    svm = SVC()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    svm.fit(X_train, y_train)
    y_pred=svm.predict(X_test)
    svm_roc=roc_auc_score(y_test, y_pred)
    svm_recall=recall_score(y_test,y_pred)
    svm_precision=precision_score(y_test,y_pred)
    return accuracy_score(y_test,y_pred),svm_roc,svm_recall,svm_precision

def rf_model(X, y):
    rf = RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    rf.fit(X_train, y_train)
    y_pred=rf.predict(X_test)
    svm_roc=roc_auc_score(y_test, y_pred)
    svm_recall=recall_score(y_test,y_pred)
    svm_precision=precision_score(y_test,y_pred)
    return accuracy_score(y_test,y_pred),svm_roc,svm_recall,svm_precision


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form['model_name']
    algorithm_number=int(request.form['algorithm_number'])
    X, y = load_data()
    if algorithm_number==1:
        if model_name == 'knn':
            accuracy,roc,recall,precison = knn_model(X, y)
        elif model_name == 'lr':
            accuracy,roc,recall,precison  = lr_model(X, y)
        elif model_name == 'svm':
            accuracy,roc,recall,precison  = svm_model(X, y)
        elif model_name == 'rf':
            accuracy,roc,recall,precison  = rf_model(X, y)
        return render_template('result.html', accuracy=accuracy, model_name=model_name,roc=roc,recall=recall,precison=precison)

    if algorithm_number==2:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=5)
        X_reduced = pca.fit_transform(X_scaled)
        if model_name == 'knn':
            accuracy,roc,recall,precison = knn_model(X_reduced, y)
        elif model_name == 'lr':
            accuracy,roc,recall,precison = lr_model(X_reduced, y)
        elif model_name == 'svm':
            accuracy,roc,recall,precison = svm_model(X_reduced, y)
        elif model_name == 'rf':
            accuracy,roc,recall,precison = rf_model(X_reduced, y)
        return render_template('result.html', accuracy=accuracy, model_name=model_name,roc=roc,recall=recall,precison=precison)


    if algorithm_number==3:
        X_resampled, y_resampled = resample(X[y == 1], y[y == 1], replace=True, n_samples=X[y == 0].shape[0], random_state=42)
        X_resampled = pd.concat([X[y == 0], X_resampled])
        y_resampled = pd.concat([y[y == 0], y_resampled])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_resampled)
        if model_name == 'knn':
            accuracy,roc,recall,precison = knn_model(X_scaled, y_resampled)
        elif model_name == 'lr':
            accuracy,roc,recall,precison = lr_model(X_scaled, y_resampled)
        elif model_name == 'svm':
            accuracy,roc,recall,precison = svm_model(X_scaled, y_resampled)
        elif model_name == 'rf':
            accuracy,roc,recall,precison = rf_model(X_scaled, y_resampled)
        return render_template('result.html', accuracy=accuracy, model_name=model_name,roc=roc,recall=recall,precison=precison)

if __name__ == '__main__':
    app.run(debug=True,port=8000)
