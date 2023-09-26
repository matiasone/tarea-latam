import unittest

from fastapi.testclient import TestClient
#from challenge import app
#from api_copy import app
from challenge import api, model
import sys
sys.path.insert(0, "C:/Users/User/Desktop/'tarea latam'/challenge")
print("HOLAAAAAAAAA", api.app)

class TestBatchPipeline(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        
    def test_should_get_predict(self):
        data = {
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N", 
                    "MES": 3
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0])) # change this line to the model of chosing
        response = self.client.post("/predict", data)
        print("RESP", response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predict": [0]})
    

    def test_should_failed_unkown_column_1(self):
        data = {       
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N",
                    "MES": 13
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))# change this line to the model of chosing
        response = self.client.post("/predict", data)
        print("RESP", response)
        self.assertEqual(response.status_code, 400)

    def test_should_failed_unkown_column_2(self):
        data = {        
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "O", 
                    "MES": 13
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))# change this line to the model of chosing
        response = self.client.post("/predict", data)
        print("RESP", response)
        self.assertEqual(response.status_code, 400)
    
    def test_should_failed_unkown_column_3(self):
        data = {        
            "flights": [
                {
                    "OPERA": "Argentinas", 
                    "TIPOVUELO": "O", 
                    "MES": 13
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))
        response = self.client.post("/predict", data)
        print("RESP", response)
        self.assertEqual(response.status_code, 400)