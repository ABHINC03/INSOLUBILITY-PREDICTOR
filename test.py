import pickle
import joblib
import pandas as pd
from cal import get_molecular_features,calculate_aromatic_proportion
with open ('C:\PROJECT\ISOL\solubility_model.pkl','rb') as file:
    model=pickle.load(file)
Features=joblib.load('C:\PROJECT\ISOL\solubilty_features.pkl')

data=get_molecular_features('CC(=O)Nc1ccc(O)cc1')
print(data)
df=pd.DataFrame([data])
X=df.iloc[:,:-1]
print(X.columns)
print(X)