# importando as bibliotecas
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas_profiling

%matplotlib inline
tqdm.pandas(desc="Operation Progress")

test_file = pd.read_csv("train.csv")
test_file = test_file.drop('ID_code',axis=1)
test_sample = test_file.sample(n=384,random_state=1)

# Definindo código para retornar features mais correlatas
# https://towardsdatascience.com/feature-selection-correlation-and-p-value-da8921bfb3cf
# Fazer df com amostra de 384
# Rodar os códigos abaixo e gerar um modelo fitando maiores correlações e rodando o predict

def get_redundant_pairs(test_sample):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = test_sample.columns
    for i in range(0, test_file.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(test_sample, n=5):
    au_corr = test_file.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(test_sample)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(test_sample, 10))

# dividindo o dataset em features e variável target
from sklearn.model_selection import train_test_split
feature_cols = test_sample['target']

# Features
X = test_sample.iloc[:,1:201]
# Target variable
y = np.array(test_sample[['target']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = y_train.reshape((307,))
y_test = y_test.reshape((77,))

X=np.matrix(test_sample.iloc[:,1:201])
y=np.matrix(test_sample[['target']])

from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier

lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

embeded_lgb_selector = SelectFromModel(lgbc, max_features=199)
embeded_lgb_selector.fit(X, y)

embeded_lgb_support = embeded_lgb_selector.get_support()
embeded_lgb_feature = test_sample.loc[:,embeded_lgb_support].columns.tolist()
print(str(len(embeded_lgb_feature)), 'selected features')

embeded_lgb_feature.remove('target')
embeded_lgb_feature

# importando as bibliotecas dos modelos classificadores
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# definindo uma lista com todos os modelos
classifiers = [
    KNeighborsClassifier(3),
    GaussianNB(),
    LogisticRegression(),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier()]

# rotina para instanciar, predizer e medir os rasultados de todos os modelos
for clf in classifiers:
    # instanciando o modelo
    clf.fit(X_train, y_train)
    # armazenando o nome do modelo na variável name
    name = clf.__class__.__name__
    # imprimindo o nome do modelo
    print("="*30)
    print(name)
    # imprimindo os resultados do modelo
    print('****Results****')
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))

# Rodando o modelo no df_test

df_test = pd.read_csv('C:/Users/rco1/Desktop/Digital House/3. Machine Learning Introdutorio/99. Desafio 3/test.csv')

test_sample = df_test.sample(n=384,random_state=1)

def get_redundant_pairs(test_sample):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = test_sample.columns
    for i in range(0, test_file.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(test_sample, n=5):
    au_corr = test_file.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(test_sample)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

# Features
X = test_sample.iloc[:,1:201]
# Target variable
y = np.array(test_sample[['target']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = RandomForestClassifier().predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))

# submission
predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)
predictions.to_csv('lgb_all_predictions.csv', index=None)
sub_df = pd.DataFrame({"ID_code":df_test["ID_code"].values})
sub_df["target"] = predictions['target']
sub_df.to_csv("lgb_submission.csv", index=False)
oof.to_csv('lgb_oof.csv', index=False)
