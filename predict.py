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
        dataframe.to_csv(args.output, sep = ';', decimal = '.')
    else:
        print("Es necesario proporcionar un archivo con un modelo para realizar una predicción")