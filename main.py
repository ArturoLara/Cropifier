from model import Model
import pandas as pd

import tools

if __name__ == '__main__':

    args = tools.get_args()
    model = Model()

    x_data = pd.read_csv(args.x_dataset, sep = ';', decimal = '.', index_col='id')

    if args.model_file: #predict
        model.load_model(args.model_file)
        prediction = model.predict(x_data)
        dataframe = pd.DataFrame(prediction, columns=['Crop'], index=x_data.index)
        dataframe.to_csv(args.output, sep = ';', decimal = '.')#, index_col='id')
    elif args.y_dataset: #fit
        y_data = pd.read_csv(args.y_dataset, sep = ';', decimal = '.', index_col='id')
        score = model.fit(x_data, y_data, args.output)
        print("Score obtenido tras el entrenamiento: ", score)
        c_matrix, class_names = model.export_conf_matrix(model.predict(x_data), y_data)
        tools.confusion_matrix(c_matrix, class_names)

        # competition score
        classes_repetitions = y_data.value_counts()
        cm_diag = c_matrix.diagonal(0)
        competition_score = cm_diag[2]/classes_repetitions['corn'] + cm_diag[0]/classes_repetitions['soybean'] + cm_diag[3]/classes_repetitions['cotton'] + cm_diag[4]/classes_repetitions['rice'] + cm_diag[1]/classes_repetitions['winter_wheat']
        print(competition_score)

        if args.out_matrix:
            c_matrix, class_names = model.export_conf_matrix(model.predict(x_data), y_data)
            tools.confusion_matrix(c_matrix, class_names, args.out_matrix)
