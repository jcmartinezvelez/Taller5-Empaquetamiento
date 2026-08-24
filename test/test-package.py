from pathlib import Path

import pandas as pd

from model.predict import make_prediction


data_file = Path(__file__).resolve().parent / "bankchurn_test.csv"

sample_input_data = pd.read_csv(data_file)

result = make_prediction(input_data=sample_input_data)

predictions = result.get("predictions") or []

print("Versión del modelo:", result.get("version"))
print("Número de predicciones:", len(predictions))
print("Primeras 10 predicciones:", predictions[:10])
print("Errores:", result.get("errors"))
