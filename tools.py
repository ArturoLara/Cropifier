import argparse
import matplotlib.pyplot as plt
import pandas as pd

#read Args
def get_args():
    parser = argparse.ArgumentParser(description='Process some machine learning tasks.')

    subparsers = parser.add_subparsers(dest='command', required=True)

    # Subparser for the 'fit' command
    parser_fit = subparsers.add_parser('fit', help='Fit the model with the given datasets.')
    parser_fit.add_argument('X_dataset', type=str, help='Path to the input features dataset.')
    parser_fit.add_argument('Y_dataset', type=str, help='Path to the target values dataset.')
    parser_fit.add_argument('output', type=str, help='Path to save the fitted model.')

    # Subparser for the 'predict' command
    parser_predict = subparsers.add_parser('predict', help='Predict using the given model and dataset.')
    parser_predict.add_argument('model_file', type=str, help='Path to the model file.')
    parser_predict.add_argument('X_dataset', type=str, help='Path to the input features dataset.')
    parser_predict.add_argument('output', type=str, help='Path to save the predictions.')

    # Subparser for the 'score' command
    parser_score = subparsers.add_parser('score', help='Score the model using the given datasets.')
    parser_score.add_argument('model_file', type=str, help='Path to the model file.')
    parser_score.add_argument('X_dataset', type=str, help='Path to the input features dataset.')
    parser_score.add_argument('Y_dataset', type=str, help='Path to the target values dataset.')
    parser_score.add_argument('--output', type=str, help='Optional path to save the confusion matrix results.')

    args = parser.parse_args()
    return args

#save plot
def save_confusion_matrix(matrix, class_names, output):
    # Visualizar la matriz de confusión con matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation='nearest', cmap='Blues')
    plt.title("Matriz de Confusión")
    plt.colorbar()

    # Etiquetas en los ejes
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Anotaciones en la matriz de confusión
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, matrix[i, j], ha="center", va="center", color="red")

    plt.ylabel("Etiqueta real")
    plt.xlabel("Predicción")
    plt.tight_layout()
    plt.savefig(output)

#load CSV
def load_csv(csv_path, separator = ';', decimal = '.', index = 'id'):

    data = pd.read_csv(csv_path, sep=separator, decimal=decimal, index_col=index)
    return data

#save CSV
def export_csv(output, data, separator = ';', decimal = '.', index = 'id'):

    data.to_csv(output, sep=separator, decimal=decimal, index_col=index)