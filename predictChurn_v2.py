import sys
import pandas as pd
from pycaret.classification import predict_model, load_model

class ChurnPredictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, df):
        predictions = predict_model(self.model, data=df)

        predictions.rename(
            {'prediction_label': 'Churn_prediction'},
            axis=1,
            inplace=True
        )

        predictions['Churn_prediction'].replace(
            {1: 'Churn', 0: 'No churn'},
            inplace=True
        )

        return predictions['Churn_prediction']
    
    def score_predict(self, df):
        predictions = predict_model(self.model, data=df)

        predictions.rename(
            {'prediction_label': 'Churn_prediction', 'prediction_score': 'Churn_probability'},
            axis=1,
            inplace=True
        )

        predictions['Churn_prediction'] = predictions['Churn_prediction'].replace(
            {1: 'Churn', 0: 'No churn'}
        )

        return predictions[['Churn_prediction', 'Churn_probability']]


def load_data(filepath):
    return pd.read_csv(filepath, index_col='customerID')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predictChurn.py <data_file> <model_path>")
        sys.exit(1)

    data_file = sys.argv[1]
    model_path = sys.argv[2]

    #df = load_data('Data/new_churn_data.csv')
    df = load_data(data_file)

    predictor = ChurnPredictor(model_path)

    if "Ridge" in model_path or "SVM" in model_path:
        predictions = predictor.predict(df)
    else:
        predictions = predictor.score_predict(df)

    print("predictions:")
    print(predictions)
