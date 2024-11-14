from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import  SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import joblib

class Model:

    __pipe = Pipeline
    __var_th = .02

    def __init__(self):
        input_preprocess = []

        # eliminamos los datos que no varían casi (no dan mucha información)
        unit_scaler = MinMaxScaler().set_output(transform="pandas")
        input_preprocess.append(('scaler1', unit_scaler))
        feat_selector = VarianceThreshold(self.__var_th).set_output(transform='pandas')
        input_preprocess.append(('varthresh', feat_selector))

        preprocess_pipe = Pipeline(input_preprocess)

        voting_system = 'soft' #<-- choices are 'soft' , 'hard'

        lin_clf  = LogisticRegression(max_iter=1000)
        svm_clf  = SVC(kernel='poly', degree=8,probability=True)
        rf_clf = RandomForestClassifier(n_estimators=1000)
        nbc_clf  = GaussianNB()

        voting_clf = VotingClassifier(
            estimators=[('lin', lin_clf),  ('svc', svm_clf), ('rf', rf_clf), ('nbc',nbc_clf)],
            voting=voting_system)

        classifier_pipe = Pipeline([('voting_clf', voting_clf)])

        self.__pipe = Pipeline([('preprocess', preprocess_pipe), ('classifier', classifier_pipe)])

    def load_model(self, model):
        self.__pipe = joblib.load(model)

    def fit(self, x_dataset, y_dataset, output):

        sm = SMOTE(random_state=42)
        x_res, y_res = sm.fit_resample(x_dataset, y_dataset)
        x_train_init, x_test_init, y_train_init, y_test_init = train_test_split(x_res, y_res, test_size=0.2, random_state=1234)

        self.__pipe.fit(x_train_init, y_train_init[['Crop']].values.ravel())
        self.__export_model(output)

        return self.__score(x_test_init, y_test_init)

    def predict(self, x_dataset):
        return self.__pipe.predict(x_dataset)

    def export_conf_matrix(self, predictions, targets):
        class_names = targets['Crop'].unique()
        confusion_m = confusion_matrix(targets, predictions, labels=class_names)

        return confusion_m, class_names

    def __score(self, x_dataset, y_dataset):
        score = self.__pipe.score(x_dataset, y_dataset[['Crop']].values.ravel())
        return score

    def __export_model(self, output):
        return joblib.dump(self.__pipe, output)
