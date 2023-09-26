from model import DelayModel, top_10_features, LR_model #, XGBoost_model
import pandas as pd
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
data = {
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N", 
                    "MES": 3
                },
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N", 
                    "MES": 8
                }
            ]
        }
ml_model = LR_model
vuelos = data["flights"]
list_to_return = []
for i in vuelos:
    dict_to_use = {}
    for key in i:
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