import pandas as pd
import numpy as np
from typing import Tuple, Union, List
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report, fbeta_score, f1_score
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from xgboost import plot_importance
from sklearn.utils.validation import check_is_fitted

def get_period_day(date):
    date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
    morning_min = datetime.strptime("05:00", '%H:%M').time()
    morning_max = datetime.strptime("11:59", '%H:%M').time()
    afternoon_min = datetime.strptime("12:00", '%H:%M').time()
    afternoon_max = datetime.strptime("18:59", '%H:%M').time()
    evening_min = datetime.strptime("19:00", '%H:%M').time()
    evening_max = datetime.strptime("23:59", '%H:%M').time()
    night_min = datetime.strptime("00:00", '%H:%M').time()
    night_max = datetime.strptime("4:59", '%H:%M').time()
    # print(date, date_time, morning_min, morning_max, afternoon_min, afternoon_max, evening_min, evening_max, night_min, night_max)
    
    if(date_time > morning_min and date_time < morning_max):
        return 'mañana'
    elif(date_time > afternoon_min and date_time < afternoon_max):
        return 'tarde'
    elif(
        (date_time > evening_min and date_time < evening_max) or
        (date_time > night_min and date_time < night_max)
    ):
        return 'noche'

def is_high_season(fecha):
    fecha_año = int(fecha.split('-')[0])
    fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = fecha_año)
    range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year = fecha_año)
    range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year = fecha_año)
    range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year = fecha_año)
    range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year = fecha_año)
    range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year = fecha_año)
    range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year = fecha_año)
    range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year = fecha_año)
    
    if ((fecha >= range1_min and fecha <= range1_max) or 
        (fecha >= range2_min and fecha <= range2_max) or 
        (fecha >= range3_min and fecha <= range3_max) or
        (fecha >= range4_min and fecha <= range4_max)):
        return 1
    else:
        return 0

def get_min_diff(data):
    fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    min_diff = ((fecha_o - fecha_i).total_seconds())/60
    return min_diff

top_10_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]

class DelayModel:

    def __init__(self):
        #self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale=4.44) # falta la escala
        #self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
        self._model = LogisticRegression()
        self.data = 0
    
    def preprocess(self, data: pd.DataFrame, target_column: str = None
                   ):
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """

        data['period_day'] = data['Fecha-I'].apply(get_period_day)
        data['high_season'] = data['Fecha-I'].apply(is_high_season)
        data['min_diff'] = data.apply(get_min_diff, axis = 1)
        threshold_in_minutes = 15
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
        self.data = data
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix = 'MES')], 
            axis = 1
        )
        top_10_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        features = features[top_10_features]
        target = data['delay']
        target = pd.Series(target, name="delay")
        target = target.to_frame()
        if target_column == None:
            return features
        elif isinstance(target_column, str):
            return (features, target)

    def fit(self, features: pd.DataFrame, target: pd.DataFrame
            ):
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        features = features.squeeze()
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.33, random_state = 42)
        ceros = 0; unos = 0
        for i in y_train["delay"]:
            if i == 0:
                ceros += 1
            elif i == 1:
                unos += 1
        scale = float(ceros/unos)
        self._model = LogisticRegression(class_weight={1: ceros/len(y_train), 0: unos/len(y_train)})
        # self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight = scale)
        try:
            print(check_is_fitted(self._model, attributes=None))
        except:
            self._model.fit(x_train, y_train)
            print(check_is_fitted(self._model, attributes=None))

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        """if not self._model.__sklearn_is_fitted__():
            all_features = self.preprocess(self.data, "delay")
            xs = all_features[0]; ys = all_features[1]
            self.fit(xs, ys)"""
        try:
            print(check_is_fitted(self._model, attributes=None))
        except:
            all_features = self.preprocess(self.data, "delay")
            xs = all_features[0]; ys = all_features[1]
            self.fit(xs, ys)
            print(check_is_fitted(self._model, attributes=None))
        
        y_preds = self._model.predict(features)
        y_preds = y_preds.tolist()
        return y_preds


data = pd.read_csv("./data/data.csv")
"""XGBoost_model = DelayModel()
features = XGBoost_model.preprocess(data, "delay")
xs = features[0]; ys = features[1]
XGBoost_model.fit(xs, ys)
lista_ys = XGBoost_model.predict(xs)
report = classification_report(ys, lista_ys, output_dict=True)
# print(report)"""

LR_model = DelayModel()
features1 = LR_model.preprocess(data, "delay")
xs1 = features1[0]; ys1 = features1[1]
LR_model.fit(xs1, ys1)
lista_ys1 = LR_model.predict(xs1)
report1 = classification_report(ys1, lista_ys1, output_dict=True)
print(report1)