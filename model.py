from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import  SVC

import joblib

class Model:

    __pipe = Pipeline
    __var_th = .02
    x_test = []
    y_test = []

    def __init__(self):
        input_preprocess = []


        # escalado de datos y eliminamos los datos que no varían casi (no dan mucha información)
        unit_scaler = StandardScaler().set_output(transform="pandas")
        input_preprocess.append(('scaler1', unit_scaler))
        feat_selector = VarianceThreshold(self.__var_th).set_output(transform='pandas')
        input_preprocess.append(('varthresh', feat_selector))

        preprocess_pipe = Pipeline(input_preprocess)

        svm_clf  = SVC(kernel='poly', degree=12,probability=True) #Modelo SVN
        rf_clf = RandomForestClassifier(n_estimators=1000) #Modelo Random Forest
        clf_net = MLPClassifier(solver='adam', alpha=1e-4,
                            hidden_layer_sizes=(30,40), random_state=1, max_iter=400, activation='tanh') #Modelo red MLP
        nca = NeighborhoodComponentsAnalysis(random_state=42)
        knn = KNeighborsClassifier(n_neighbors=3)
        nca_knn_pipe = Pipeline([('nca', nca), ('knn', knn)]) #Pipeline con el modelo por vecindad

        voting_clf = VotingClassifier(
            estimators=[('svm_clf', svm_clf), ('rf_clf', rf_clf), ('nca_knn', nca_knn_pipe), ('clf_net', clf_net)],
            voting='soft') #se define el sistema de votación y se incluyen los modelos

        classifier_pipe = Pipeline([('voting_clf', voting_clf)])
        self.__pipe = Pipeline([('preprocess', preprocess_pipe), ('classifier', classifier_pipe)])

    def load_model(self, model):
        self.__pipe = joblib.load(model)

    def fit(self, x_dataset, y_dataset, output):

        sm = SMOTE(random_state=42)
        x_res, y_res = sm.fit_resample(x_dataset, y_dataset) #creación de muestras sinteticas

        #Split del dataset para obtener los subconjuntos de entrenamiento y test
        x_train_init, self.x_test, y_train_init, self.y_test = train_test_split(x_res, y_res, test_size=0.01, random_state=1234)

        self.__pipe.fit(x_train_init, y_train_init[['Crop']].values.ravel())
        self.__export_model(output)

        return self.__score(self.x_test, self.y_test)

    def predict(self, x_dataset):
       return self.__pipe.predict(x_dataset)


    def export_conf_matrix(self, predictions_to_mat):
        class_names = self.y_test['Crop'].unique()
        confusion_m = confusion_matrix(self.y_test, predictions_to_mat, labels=class_names)

        return confusion_m, class_names

    def __score(self, x_dataset, y_dataset):
        score = self.__pipe.score(x_dataset, y_dataset[['Crop']].values.ravel())
        return score

    def __export_model(self, output):
        return joblib.dump(self.__pipe, output)
