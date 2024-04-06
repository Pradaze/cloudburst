import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, roc_curve, classification_report
import catboost as cb
import requests
import joblib
import time


df = pd.read_csv('https://raw.githubusercontent.com/SPUTnik-42/BadalBarsaBijuli/master/Dataset_BBB.csv')

df = df.dropna()
df = df.drop(['Date','Location','Evaporation','Rainfall','Sunshine','WindGustDirection','WindDirection9am','WindDirection3pm','WindSpeed9am','Humidity9am','Pressure9am','Cloud9am','Temperature9am','CloudBurst Today','Temperature3pm'], axis=1)

df['CloudBurstTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)
features = df[['MinimumTemperature','MaximumTemperature','WindGustSpeed','WindSpeed3pm','Humidity3pm','Pressure3pm','Cloud3pm']]
target = df['CloudBurstTomorrow']

def run_model(model, X_train, y_train, X_test, y_test, verbose=True):
    t0=time.time()
    if verbose == False:
        model.fit(X_train,y_train, verbose=0)
    else:
        model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    coh_kap = cohen_kappa_score(y_test, y_pred)
    time_taken = time.time()-t0
    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    print("Cohen's Kappa = {}".format(coh_kap))
    print("Time taken = {}".format(time_taken))
    print(classification_report(y_test,y_pred,digits=5))

    probs = model.predict_proba(X_test)
    probs = probs[:, 1]
    fper, tper, thresholds = roc_curve(y_test, probs)



    return model, accuracy, roc_auc, coh_kap, time_taken

params_cb ={'iterations': 50,
            'max_depth': 16}
model_cb = cb.CatBoostClassifier(**params_cb)
model_cb, accuracy_cb, roc_auc_cb, coh_kap_cb, tt_cb = run_model(model_cb, X_train, y_train, X_test, y_test, verbose=False)

city_name = "Dehradun"
api_key = '5e5d431b17906d316dc226222b1047aa'
url = f'http://api.openweathermap.org/data/2.5/weather?q={city_name}&units=metric&appid={api_key}'
response = requests.get(url)
data = response.json()

temp_min = data['main']['temp_min']
temp_max = data['main']['temp_max']
humidity = data['main']['humidity']
pressure = data['main']['pressure']
wind_speed = data['wind']['speed']
wind_gust = data['wind']['gust']
clouds = data['clouds']['all']

print(temp_min,temp_max,humidity,pressure,wind_speed,wind_gust,clouds)
input_data = pd.DataFrame({
    'MinimumTemperature': [temp_min],
    'MaximumTemperature': [temp_max],
    'WindGustSpeed': wind_gust,
    'WindSpeed3pm': [wind_speed],
    'Humidity3pm': [humidity],
    'Pressure3pm': [pressure],
    'Cloud3pm': [clouds]
})
predictions = model_cb.predict(input_data)

joblib.dump(model_cb, 'model_bbb.joblib', compress=('zlib', 3))