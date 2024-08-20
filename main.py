import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

df=pd.read_excel(r'C:\Users\spano\Desktop\cleaned2.xlsx','Sheet2')
pd.set_option('display.max_columns', None)


### z-score for outlier localization
def find_outliers(col):
    from scipy import stats
    z=np.abs(stats.zscore(col))
    outliers=np.where(z>3,True,False)
    return pd.Series(outliers,index=col.index)


df_outliers=pd.DataFrame(df)
for col in df.describe().columns:
    df_outliers[col]=find_outliers(df[col])


### outliers elimination
minus_O3=df_outliers.drop(['O3'],axis=1)
outs=minus_O3.apply(lambda x: np.any(x),axis=1)
df_clean=df.loc[outs==False]


### Data scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df_scaled=scaler.fit_transform(df_clean)
X=np.delete(df_scaled,8,axis=1)
Y=df_scaled[:,8]


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=10)


from sklearn.ensemble import BaggingRegressor
bag_model=BaggingRegressor(
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
)


bag_model.fit(X_train,Y_train)
predictions=bag_model.predict(X_test)
RMSE=mean_squared_error(Y_test,predictions)
print(f"R-squared: {bag_model.score(X_test,Y_test):.2f}",f"RMSE: {RMSE:.2f}",sep='\n')












