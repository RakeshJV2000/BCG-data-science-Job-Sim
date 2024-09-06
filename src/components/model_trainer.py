import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    train_data_path=os.path.join("data","data_for_predictions.csv")
    model_path = os.path.join("data", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def model_trainer(self):
        try:
            logging.info("Entered the model training module")
            df = pd.read_csv(self.model_trainer_config.train_data_path)
            df.drop(columns=["Unnamed: 0"], inplace=True)
            df.head()

            # Make a copy of our data
            train_df = df.copy()

            # Separate target variable from independent variables
            y = df['churn']
            X = df.drop(columns=['id', 'churn'])
            print(X.shape)
            print(y.shape)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            print(X_train.shape)
            print(y_train.shape)
            print(X_test.shape)
            print(y_test.shape)

            model = RandomForestClassifier(
                n_estimators=1000
            )
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions).ravel()

            print(f"True positives: {tp}")
            print(f"False positives: {fp}")
            print(f"True negatives: {tn}")
            print(f"False negatives: {fn}\n")

            print(f"Accuracy: {metrics.accuracy_score(y_test, predictions)}")
            print(f"Precision: {metrics.precision_score(y_test, predictions)}")
            print(f"Recall: {metrics.recall_score(y_test, predictions)}")

            # Model understanding
            # A simple way of understanding the results of a model is to look at feature importance's.
            # Feature importance's indicate the importance of a feature within the predictive model,
            # there are several ways to calculate feature importance,

            feature_importance = pd.DataFrame({
                'features': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values(by='importance', ascending=True).reset_index()

            plt.figure(figsize=(15, 25))
            plt.title('Feature Importances')
            plt.barh(range(len(feature_importance)), feature_importance['importance'], color='b', align='center')
            plt.yticks(range(len(feature_importance)), feature_importance['features'])
            plt.xlabel('Importance')
            plt.savefig(os.path.join('images', "feature_importance.png"))
            plt.show()

            save_object(self.model_trainer_config.model_path, model)







        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj = ModelTrainer()
    obj.model_trainer()