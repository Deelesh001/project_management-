import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
import numpy as np

# Trained model
model = Pipeline(steps=[('preprocessor',
                 ColumnTransformer(transformers=[('num',
                                                  Pipeline(steps=[('imputer',
                                                                   SimpleImputer()),
                                                                  ('scaler',
                                                                   StandardScaler())]),
                                                  ['m (kg)', 'Enedc (g/km)',
                                                   'Ewltp (g/km)', 'ec (cm3)',
                                                   'ep (KW)',
                                                   'Fuel consumption ',
                                                   'Electric range (km)',
                                                   'Carbon Intensity '
                                                   '(gCO2/kWh)',
                                                   'Lifecycle Emissions '
                                                   '(gCO2/km)',
                                                   'Recycling Potential']),
                                                 ('cat',
                                                  Pipeline(steps=[('imputer',
                                                                   SimpleImputer(fill_value='Unknown',
                                                                                 strategy='constant')),
                                                                  ('onehot',
                                                                   OneHotEncoder(handle_unknown='ignore'))]),
                                                  ['Fuel Type', 'Cn', 'Mh'])])),
                ('regressor', GradientBoostingRegressor(random_state=42))])

# Predict function
def predict(features):
    return model.predict(features)
