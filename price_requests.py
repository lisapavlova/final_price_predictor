import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# We'll use our original data  to form requests to the web-service and compare the models' predictions to the true prices
source = pd.read_csv('cleaned_dataset.csv')
random_sample = source.sample(10)
random_sample['year'] = pd.DatetimeIndex(random_sample.first_day_exposition).year

result = pd.DataFrame(columns=['simple', 'extended', 'true'], index=[i for i in range(0, 10)])

model_versions = ['simple', 'extended']

for i in range(0, 10):
    payload = {'floor': random_sample.iloc[i]['floor'],
               'open_plan': random_sample.iloc[i]['open_plan'],
               'rooms': random_sample.iloc[i]['rooms'],
               'studio': random_sample.iloc[i]['studio'],
               'area': random_sample.iloc[i]['area'],
               'kitchen_area': random_sample.iloc[i]['kitchen_area'],
               'living_area': random_sample.iloc[i]['living_area'],
               'renovation': int(random_sample.iloc[i]['renovation']),
               'building_id': random_sample.iloc[i]['building_id'],
               'year': random_sample.iloc[i]['year']}
    for mv in model_versions:
        payload['model_version'] = mv
        r = requests.get('http://51.250.102.58:5444/predict_price', params=payload)
        result.loc[i, mv] = float(r.text)
    result.loc[i, 'true'] = random_sample.iloc[i]['last_price']

# Here we'll output a chart that will visually depict the difference between the models, and between true values and predictions
# NOTE: The chart should open in console after running the code; make sure the web-service is active
sns.scatterplot(data=result)
plt.show()