from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import pandas as pd

def load_data(num_records):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
    data = pd.read_csv(url)
    data.drop('name', axis=1, inplace=True)
    X = data.drop('status', axis=1)
    y = data['status']
    return X[:num_records], y[:num_records]

def knn_model(X, y):
    knn = KNeighborsClassifier()
    knn.fit(X, y)
    return knn

def lr_model(X, y):
    lr = LogisticRegression()
    lr.fit(X, y)
    return lr

def svm_model(X, y):
    svm = SVC()
    svm.fit(X, y)
    return svm

def rf_model(X, y):
    rf = RandomForestClassifier()
    rf.fit(X, y)
    return rf

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    num_records = int(request.form['num_records'])
    model_name = request.form['model_name']
    algorithm_number=int(request.form['algorithm_number'])
    X, y = load_data(num_records)
    if algorithm_number==1:
        if model_name == 'knn':
            model = knn_model(X, y)
        elif model_name == 'lr':
            model = lr_model(X, y)
        elif model_name == 'svm':
            model = svm_model(X, y)
        elif model_name == 'rf':
            model = rf_model(X, y)
        accuracy = model.score(X, y)
        return render_template('result.html', accuracy=accuracy, model_name=model_name)

    if algorithm_number==2:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=5)
        X_reduced = pca.fit_transform(X_scaled)
        if model_name == 'knn':
            model = knn_model(X_reduced, y)
        elif model_name == 'lr':
            model = lr_model(X_reduced, y)
        elif model_name == 'svm':
            model = svm_model(X_reduced, y)
        elif model_name == 'rf':
            model = rf_model(X_reduced, y)
        accuracy = model.score(X_reduced, y)
        return render_template('result.html', accuracy=accuracy, model_name=model_name)

    if algorithm_number==3:
        X_resampled, y_resampled = resample(X[y == 1], y[y == 1], replace=True, n_samples=X[y == 0].shape[0], random_state=42)
        X_resampled = pd.concat([X[y == 0], X_resampled])
        y_resampled = pd.concat([y[y == 0], y_resampled])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_resampled)
        if model_name == 'knn':
            model = knn_model(X_scaled, y_resampled)
        elif model_name == 'lr':
            model = lr_model(X_scaled, y_resampled)
        elif model_name == 'svm':
            model = svm_model(X_scaled, y_resampled)
        elif model_name == 'rf':
            model = rf_model(X_scaled, y_resampled)
        accuracy = model.score(X_scaled,y_resampled)
        return render_template('result.html', accuracy=accuracy, model_name=model_name)

if __name__ == '__main__':
    app.run(debug=True,port=8000)
