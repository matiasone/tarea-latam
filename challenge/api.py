import fastapi
import pandas as pd
import json
from fastapi.testclient import TestClient
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

app = fastapi.FastAPI()

print(app)

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(data, response: fastapi.Response) -> dict:
    from model import DelayModel, LR_model
    ml_model = LR_model
    vuelos = data["flights"]
    list_to_return = []
    dict_to_return = {}
    for i in vuelos:
        dict_to_use = {}
        for key in i:
            if key == "MES":
                if i[key] >= 12:
                    response.status_code = 400
                    return json.dumps(dict_to_return)
            elif key == "TIPOVUELO":
                if i[key] != "I":
                    if i[key] != "N":
                        response.status_code = 400
                        return json.dumps(dict_to_return)
            combinacion = key + "_" + str(i[key])
            for feat in top_10_features:
                if feat == combinacion:
                    dict_to_use[combinacion] = [True]
        # ahora agregamos feats faltantes
        for f in top_10_features:
            if f not in dict_to_use:
                dict_to_use[f] = [False]
        df_to_use = pd.DataFrame.from_dict(dict_to_use)
        y_pred = LR_model.predict(df_to_use) # tengo que pasarle un dataframe
        print(y_pred)
        list_to_return.append(y_pred)
        print(list_to_return)
        dict_to_return = {
            'predict': y_pred
        }
        json_to_return = json.dumps(dict_to_return)
        return json_to_return
    
print("Hello World!")
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
"""client1 = TestClient(app)
response = client1.post("/predict", data)
print("RESP", response)"""