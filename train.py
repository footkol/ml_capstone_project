import pandas as pd
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from meteostat import Point, Hourly
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_error
from lightgbm import LGBMRegressor
from scipy.stats import yeojohnson
from math import exp
import pickle

start = datetime(2014, 11, 1)
end = datetime(2019, 10, 13)

brooklyn = Point(40.646081321775156, -73.95796142054905)

data = Hourly(brooklyn, start, end)
data = data.fetch()

df_weather = data.drop(columns= ['dwpt', 'snow', 'wdir', 'wpgt', 'pres', 'tsun'])

df_resampled = df_weather.resample('15T').asfreq().fillna(method='pad')

weather = df_resampled.fillna(0)
weather.isnull().sum()

df = pd.read_csv('US Holiday Dates (2004-2021).csv')

df = df.drop_duplicates(subset=['Date'])

df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M')
df['Date'] = pd.to_datetime(df['Date'])

df_intervals = pd.DataFrame()

# creating for loop to convert daily timestamps into 15 minute ones to match other databases
for date in df['Date'].dt.date.unique():
    df_day = df[df['Date'].dt.date == date]

    intervals = pd.date_range(start=df_day['Date'].min(), end=df_day['Date'].max() + pd.Timedelta(days=1), freq='15T')

    df_day_intervals = pd.DataFrame({'Date': intervals, 'holiday': 'yes'})

    df_intervals = pd.concat([df_intervals, df_day_intervals], ignore_index=True)


df_intervals.set_index('Date', inplace=True)

holidays = df_intervals

filtered_df =pd.read_csv('flatbush_avenue.csv', index_col=0)

sb_df = filtered_df[filtered_df['Direction'].isin(['sb'])]

nb_df = filtered_df[filtered_df['Direction'].isin(['nb'])]

sb_df.columns = sb_df.columns.str.lower().str.replace(' ', '_')

sb_df = sb_df.rename(columns={'yr': 'year', 'm': 'month', 'd': 'day', 'hh': 'hour', 'mm': 'minute'})
               
sb_df['date'] = pd.to_datetime(sb_df[['year', 'month', 'day', 'hour', 'minute']], format='%Y-%m-%d %H:%M')
sb_df.set_index('date', inplace=True)

refine_df = sb_df[(sb_df['fromst'] == 'brighton_line') & (sb_df['tost'] == 'brighton_line')]

clean_df = refine_df.drop(labels = ['unnamed:_0', 'requestid', 'boro', 'segmentid', 'wktgeom', 'fromst', 'tost', 'street', 'direction'], axis =1)


clean_df.head().sort_index()

duplicated = clean_df[clean_df.index.duplicated(keep=False)]
duplicated_rows = pd.DataFrame(duplicated)


agg_dict = {'vol': 'sum', 'year': 'first', 'month': 'first', 'day': 'first', 'hour': 'first', 'minute': 'first'}
combined_df = clean_df.groupby(clean_df.index).agg(agg_dict)

duplicated_clean = combined_df[combined_df.index.duplicated(keep=False)]
duplicates = pd.DataFrame(duplicated_clean)

flatbush = combined_df

df_temp = flatbush.merge(weather, left_index=True, right_index=True, how='left')
df_merged = df_temp.merge(holidays, left_index=True, right_index=True, how='left')

df = pd.get_dummies(df_merged, dtype=int)

transformed_data, lmbda = yeojohnson(df.vol)

df['vol'] = transformed_data
df.sort_index()


df['month'] = pd.to_numeric(df['month'])
df['hour'] = pd.to_numeric(df['hour'])


def generate_cyclical_features(df, col_name, period, start_num=0):
    kwargs = {
        f'sin_{col_name}' : lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
        f'cos_{col_name}' : lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)    
             }
    return df.assign(**kwargs).drop(columns=[col_name])

df_feat_hour = generate_cyclical_features(df, 'hour', 24, 0)
df_final = generate_cyclical_features(df_feat_hour, 'month', 12, 1)


df_final_cleaned = df_final.drop(columns=['year', 'day', 'minute'], axis =1)
df_final_cleaned.sort_index().head()

df_final_cleaned = df_final.sort_index()



y = df_final_cleaned[['vol']]
X = df_final_cleaned.drop(columns=['vol'])


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=True, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=1)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

y_train_scaled = scaler.fit_transform(y_train)
y_val_scaled = scaler.transform(y_val)
y_test_scaled = scaler.transform(y_test)



param_grid = {
    'num_leaves': 100, #  maximum number of leaves in one tree
    'learning_rate': 0.1, # step size at each iteration while moving toward a minimum of a loss function
    'n_estimators': 200, # number of trees in the forest
    'colsample_bytree': 0.8, # regularization technique to prevent overfitting
    'force_col_wise': True, # provides a speedup for training large datasets
    'verbosity': 0  # controls the amount of information printed during training
}

lgb_model = LGBMRegressor(objective='regression', metric='l2', boosting_type='gbdt')

best_params = param_grid
bst = LGBMRegressor(metric='l2', boosting_type='gbdt', **best_params ) 
bst.fit(X_train_scaled, y_train_scaled, eval_set=[(X_val_scaled, y_val_scaled)])
y_pred = bst.predict(X_test_scaled)


def inverse_transform(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df

def format_predictions(predictions, values, df_test, scaler):
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate([predictions.reshape(-1, 1)], axis=0).ravel() # reshaping 'predictions' into 2D array to match 'values'
    df_result = pd.DataFrame(data={"vol": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
    df_result = df_result.sort_index()
    df_result = inverse_transform(scaler, df_result, [["vol", "prediction"]])
    return df_result

df_lgbm = format_predictions(y_pred, y_test_scaled, X_test, scaler)
print(df_lgbm.head(10))


def invert_yeojhonson(value, lmbda):
  if value>= 0 and lmbda == 0:
    return exp(value) - 1
  elif value >= 0 and lmbda != 0:
    return (value * lmbda + 1) ** (1 / lmbda) - 1
  elif value < 0 and lmbda != 2:
    return 1 - (-(2 - lmbda) * value + 1) ** (1 / (2 - lmbda))
  elif value < 0 and lmbda == 2:
    return 1 - exp(-value)

df_result_lgbm = pd.DataFrame()
df_result_lgbm['vol'] = df_lgbm['vol'].apply(lambda x: invert_yeojhonson(x, lmbda))
df_result_lgbm['prediction'] = df_lgbm['prediction'].apply(lambda x: invert_yeojhonson(x, lmbda))

df_result_lgbm.head()


def calculate_metrics(df):
    return {'mse' : mean_squared_error(df.vol, df.prediction),
            'mae' : mean_absolute_error(df.vol, df.prediction),
            'rmse' : mean_squared_error(df.vol, df.prediction) ** 0.5,
            'r2' : r2_score(df.vol, df.prediction)}



calculate_metrics(df_result_lgbm)


with open('lightgbm_model.bin', 'wb') as f_out:
    pickle.dump((bst), f_out)


with open('scaler_model.bin', 'wb') as f_out:
    pickle.dump((scaler, lmbda), f_out)

