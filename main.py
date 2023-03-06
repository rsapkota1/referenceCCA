import numpy as np
from customrCCA import SCCA_IPLS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import random
import matplotlib.pyplot as plt
from cca_zoo.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


df_pca_gray=pd.read_csv('/data/users2/rsapkota/SCCA/Gray_Matter/PCA__components_gray.csv').iloc[:,1:]

df_pca_white=pd.read_csv('/data/users2/rsapkota/SCCA/Gray_Matter/PCA__components_white.csv').iloc[:,1:]

df_cognition=pd.read_csv('/data/users2/rsapkota/SCCA/Gray_Matter/Cognition_Uncorrected_Subtract_Result_data_all_replace_nan_zero_normalized.csv')['nihtbx_picvocab_uncorrected']

df_cognition = np.nan_to_num(df_cognition, nan=0)
X_train, X_test, y_train, y_test, ref_train, ref_test = train_test_split(df_pca_gray, df_pca_white, df_cognition, test_size=0.3, random_state=50)

ref_train = np.array(ref_train).reshape(-1,1)
ref_test = np.array(ref_test).reshape(-1,1)


def scorer(estimator,X):
    dim_corrs=estimator.score(X) #Goes inside _basemodel.py score function
    return dim_corrs.mean()

c1 = [0.1,0.01]
c2= [0.1,0.01]
c3 = [0.4]#Optional
param_grid = {'c': [c1,c2,c3]}
lrs = [0.01]
cv = 2 #number of folds


#-----USAGE without GridSearch - UNCOMMENT TO USE---------------------------------------------------
# model=SCCA_IPLS(c=[0.01,0.01,0.001], latent_dims=3,verbose=True, lr=0.001)
# referenceCCAModel=model.fit([X_train,y_train, ref_train])
# train_scores = referenceCCAModel.score([X_train, y_train, ref_train])
# test_scores = referenceCCAModel.score([X_test, y_test, ref_test])

#------END-----------------------------------------------------------------------


#-----USAGE with GridSearch---------------------------------------------------
model=GridSearchCV(SCCA_IPLS(latent_dims=3, verbose=True, lr=0.001, reference=ref_train),param_grid=param_grid,scoring=scorer)
referenceCCAModel = model.fit([X_train,y_train, ref_train])
train_scores = referenceCCAModel.best_estimator_.score([X_train,y_train, ref_train]) ##Goes inside _basemodel.py score function
test_scores=referenceCCAModel.best_estimator_.score([X_test, y_test, ref_test]) ##Goes inside _basemodel.py score function
#------END-----------------------------------------------------------------------





print(train_scores)
print(test_scores)







