import sys
import pandas as pd
from scipy.stats import percentileofscore
from pycaret.classification import predict_model, load_model

class ChurnPredictor:
    def __init__(self, model_path, train_path=None):
        self.model = load_model(model_path)
        self.train_probability = None
        if train_path:
            train_df = self.load_data(train_path)
            train_df = train_df.drop(columns=['Churn', 'AvgMonthly'], errors='ignore')
            self.train_probability = self.get_churn_probability(train_df)

    def _predict(self, df):
        df = df.drop(columns=['Churn'], errors='ignore')
        return predict_model(self.model, data=df, raw_score=True)

    @staticmethod
    def load_data(filepath):
        df = pd.read_csv(filepath, index_col='customerID')
        return df

    #def get_churn_probability(self, df):
    #    probability = self.model.predict_proba(df)
    #    classes = list(self.model.classes_)
    #    # position of the churn class (1 or 'Yes'), defaulting to the last column
    #    idx = next((classes.index(c) for c in (1, 'Yes', 'Churn', True) if c in classes), -1)
    #    return pd.Series(probability[:, idx], index=df.index)

    def get_churn_probability(self, df):
        preds = self._predict(df)
        pos = ('1', 'yes', 'churn', 'true')

        for c in preds.columns:
            if c.startswith('prediction_score_') and c[len('prediction_score_'):].lower() in pos:
                return preds[c]

        # Fallback: prediction_score is the probability of the predicted class
        is_pos = preds['prediction_label'].astype(str).str.lower().isin(pos)
        score = preds['prediction_score']
        return score.where(is_pos, 1 - score)

    def make_predictions(self, df):#, model_path):
        #model = load_model('Models/GBC')
        probability = self.get_churn_probability(df)
        labels = predict_model(self.model, data=df)['prediction_label']
        labels = labels.replace({1: 'Churn', 0: 'No Churn', 'Yes': 'Churn', 'No': 'No Churn'})

        results = pd.DataFrame({
            'Churn_prediction': labels,
            'Churn_probability': probability,#.round(3),
        })
        if self.train_probability is not None:
            results['Percentile'] = probability.apply(
                lambda p: percentileofscore(self.train_probability, p)
            )#.round(1)
        return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predictChurn.py <data_file> <model_path> [training_data_file]")
        sys.exit(1)

    #train_df = load_data(sys.argv[3])
    train_path = sys.argv[3] if len(sys.argv) > 3 else None
    predictor = ChurnPredictor(model_path=sys.argv[2], train_path=train_path)
    
    #df = load_data('Data/new_churn_data.csv')
    df = predictor.load_data(sys.argv[1])
    
    predictions = predictor.make_predictions(df)#, sys.argv[2])
    #predictions = make_predictions(df, sys.argv[2], sys.argv[3])
    print('predictions:')
    print(predictions)
