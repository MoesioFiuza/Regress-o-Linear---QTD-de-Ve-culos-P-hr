import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

planilha = r'C:\Users\moesios\Desktop\EXTERNAS\regressão python.xlsx'

df_centrob = pd.read_excel(planilha, sheet_name='Centro-bairro')
df_bairroc = pd.read_excel(planilha, sheet_name='bairro-centro')

def preparar_e_prever(df, ajuste_fator=0.4):
    df['Minutos'] = df['Intervalo'].apply(lambda x: x.hour * 60 + x.minute)
    columns_of_interest = ['Vol 15min', 'Vol 60min', 'Pedestres', 'Bicicletas', 'Motocicletas', 'Automóveis', 'Ônibus', 'Caminhões']

    X = df['Minutos'].values.reshape(-1, 1)
    predictions = {}

    for column in columns_of_interest:
        if column in df.columns:
            df[column] = df[column].replace('--', np.nan).fillna(0).astype(float)
            y = df[column].values

            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)

            model = LinearRegression()
            model.fit(X_poly, y)

            future_times = np.arange(0, 285, 15).reshape(-1, 1)
            future_times_poly = poly.transform(future_times)
            future_predictions = model.predict(future_times_poly)

        
            min_val, max_val = y.min(), y.max()
            future_predictions = np.clip(future_predictions, min_val, max_val)

            future_predictions_adjusted = future_predictions * ajuste_fator

            noise = np.random.uniform(-0.5, 0.5, future_predictions_adjusted.shape)
            future_predictions_adjusted = future_predictions_adjusted + noise

            predictions[column] = np.round(future_predictions_adjusted).astype(int)

    intervals = pd.date_range('00:00', '04:45', freq='15min').time
    result_df = pd.DataFrame(predictions, index=intervals[:len(next(iter(predictions.values())))])
    result_df.index.name = 'Intervalo'

    return result_df

ajuste_fator = 0.4

result_centrob = preparar_e_prever(df_centrob, ajuste_fator)
result_bairroc = preparar_e_prever(df_bairroc, ajuste_fator)

output_path = r'C:\Users\moesios\Desktop\EXTERNAS\saida.xlsx'
with pd.ExcelWriter(output_path) as writer:
    result_centrob.to_excel(writer, sheet_name='Centro-bairro')
    result_bairroc.to_excel(writer, sheet_name='bairro-centro')

