# Flask Web-Service for predicting apartment rent prices
##### Final Project for _End-End Sample ML Project_ course

Links to repositories: [github]; [dockerhub]

This is a repository containing source code for a Flask web-service which, through the means of Machine Learning, _predicts the rent price of an apartment_ in Saint-Petersburg or its suburbs based on its parameters.

##### Contents of the repository:
- **mlmodels** folder: contains pickle files with ML predictive models and scalers for features and target variables;
- **.dockerignore**: specifies files that Docker should ignore;
- **.gitignore**: specifies intentionally untracked files that Git should ignore;
- **Dockerfile**: contains commands for assembling a Docker image;
- **app\.py**: contains source code in Python for the web-service;
- **requirements.txt**: frozen modules requirements needed for the code to function correctly;
- **price_requests.py**: python script for testing web-service with requests.

### 1. Source data and some statistics
Our source data comes from Yandex's classified service [Yandex.Realty] and contains real estate listings for flats in Saint-Petersburg and Leningradskaya Oblast.
It covers the period from 2016 till the middle of August, 2018.

Here's a breafdown of souce dataset's content:
| Variable Name | Initial Data Type | Type of Variable |
| :-----------: | :-----------: | :-----------: |
| offer_id   | int64  | feature  |
| first_day_exposition       | object        |   feature    |
| last_day_exposition      | object       | feature       |
| **last_price**       | **float64**        | **TARGET**      |
| floor       | int64        | feature      |
| open_plan       | bool        | feature      |
| rooms       | int64        | feature      |
| studio       | bool       | feature     |
| area     | float64        | feature       |
| kitchen_area      | float64        | feature      |
| living_area	      | float64        | feature   |
| agent_fee     | float64       | feature      |
| renovation      | float64        | feature     |
| offer_type     | int64        | feature      |
| category_type     | object        | feature    |
| unified_address	     | object        | feature   |
| building_id    | object        | feature   |
_Note_: 'object' type denotes strings.

Here are charts depicting relationships between features and their relationships with target variable:
![correlation heatmap](/images/corr.png) 
![scatter plots](/images/scatterplots.png) 

### 2. Models and chosen framework
Two machine learning models were created. Both were CatBoost models, one denoted 'simple' for a limited number of features used, and another - 'extended', as it employed more features, some of them calculated from initial ones.

| Model | Test MAE | Test MSE | Test RMSE |
| :-----------: | :-----------: | :-----------: | :-----------: |
| 'Simple'  | 0.3080  | 0.2906  | 0.5391 |
| 'Extended'       | 0.2836        | 0.2630       | 0.5128 |

Here's a chunk of code defining our learning and evaluation pools and our model.The best model was chosen according to bestTest result, in our case the lowest RMSE for test sample.
```python
learn_pool = Pool(X_train_cb, y_train_cb, cat_features=categorical_features, feature_names=list(X_train_cb))
test_pool = Pool(X_test_cb, y_test_cb, cat_features=categorical_features, feature_names=list(X_test_cb))
cb = CatBoostRegressor(iterations=1000, learning_rate=0.1,
                        random_state=0, eval_metric='RMSE', loss_function='RMSE')
cb.fit(learn_pool, eval_set=test_pool, verbose=100)
```

Through several attempts the better result was achieved on a `learning_rate=0.1`

Categorical features for __'extended'__ model were: _'renovation', 'building_id', 'floor'_. For the __'simple'__ model - only  _'renovation' and 'floors'_, without the _'building_id'_, which was not used at all in this model.

Features used in __Simple model__:
- floor
- open_plan
- rooms
- studio
- area
- kitchen_area
- living_area
- renovation


Features used in __Extended model__:
- floor
- open_plan
- rooms
- studio
- area
- kitchen_area
- living_area
- renovation
- building_id
- year

### 3. How to install instructions and run app with virtual environment
First, create a virtual environment in the local repository you're planning to deploy the solution, with:
```bash
sudo apt install python3.8-venv
python3 -m venv env
source env/bin/activate
```
where `env` is the name of your virtual environment.

Next, run:
```bash
pip3 install -r requirements.txt
```
to install all required packages, listed in __requirements.txt__ file.

Finally, run the python script:
```bash
python3 app.py
```
### 4. Dockerfile and its content
```Dockerfile
from ubuntu:20.04
MAINTAINER Elizaveta Pavlova
RUN apt-get update -y && \
	apt install -y python3-pip
COPY requirements.txt requirements.txt 
RUN pip3 install -r requirements.txt
COPY . .
CMD python3 app.py
```
### 5. How to open the port in your remote VM?
First, log into your Virtual Machine through terminal with 
`ssh your_username@your_VM_publicIP`
Then run `sudo ufw allow 5444` to open the necessary port, where 5444 stands for the port used in app, and `sudo` provides admin rights.

### 6. How to run app using docker and which port it uses?
First, pull the appropriate tag  from the DockerHub (latest image version is v.0.2):
```bash
docker pull laskovey/final_price_predictor:v.0.2
```
Then run the image:
```bash
docker run --network host -d laskovey/final_price_predictor:v.0.2
```
_Note_: Option `-d` (`--detach`) runs container in background and prints container ID.

The application inside the container will be accessed using port __5444__ at the host's IP address.


### 7*. Testing web-service with requests
File **price_requests.py** contains python script for testing web-service with requests. It forms requests from the our cleaned dataset (dataset itself is not on GitHub) and sends request to the service, collecting the predictions for each instance. It then outputs the chart (scatterplot) showcasing the difference of 2 models predictions for one apartment, and difference between them and the true value from the dataset.

Here's a result of comparison of predictions for requests for 10 random apartments:
![model comparison](/images/model_comparison.png) 

And the result is not as satisfying as was initially expected 👎
Athough the test RMSE seemed good while building the model, it can be seen from chart that the predictions for most instances are rather far from the true price values.
On the bright side, it seems that both 'simple' and 'extended' models catch the general tendency rather well ✊







[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [Yandex.Realty]: <https://realty.yandex.ru>
   [github]: <https://github.com/lisapavlova/final_price_predictor>
   [dockerhub]: <https://hub.docker.com/repository/docker/laskovey/final_price_predictor>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
