import argparse
import matplotlib.pyplot as plt
import pandas as pd

#read Args
def get_args():
    parser = argparse.ArgumentParser(description='Process some machine learning tasks.')

    parser.add_argument('-m', '--model_file', type=str, required=False, help='Needed to predict: Path to the model file.')
    parser.add_argument('-x','--x_dataset', type=str, required=True, help='Path to the input features dataset.')
    parser.add_argument('-y','--y_dataset', type=str, required=False, help='Needed to fit: Path to the target values dataset.')
    parser.add_argument('-o','--output', type=str, required=True, help='Path to save the predictions or the model.')
    parser.add_argument('-om','--out_matrix', type=str, required=False, help='Optional path to save the confusion matrix results.')

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