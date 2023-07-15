#import library yang diperlukan

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

#loading data ke dataframe 

data = pd.read_csv("phishing.csv")
data.head()

#share data frame 
data.shape

#listing featur of dataset
data.columns

#informasi tentang dataset
data.info()

#unique value in columns
data.nunique()

#droping index columns
data = data.drop(['Index'], axis = 1)

#deskripsi of dataset
data.describe().T

#correlation heatmap

plt.figure(figsize=(15,15))
sns.heatmap(data.corr(), annot = True)
plt.show()

#pairplot untuk fitur tertentu

df = data[['Symbol@', 'SubDomains', 'HTTPS', 'AnchorURL', 'WebsiteTraffic', 'class']]
sns.pairplot(data = df, hue="class", corner=True)

#phishing hitung dalam diagram lingkaran
data['class'].value_counts().plot(kind='pie', autopct='%1.2f%%')
plt.title("Phishing URL Count")
plt.show()

#memisah dataset menjadi fitur dependen dan independen

X = data.drop(["class"], axis = 1)
y = data["class"]

#membagi dataset menjadi set train dan test : 80-20
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape

#bikin holer untuk menyimpan performa model
ML_Model = []
accuracy = []
f1_score = []
recall = []
precision = []

#fungsi untuk memanggil dan menyimpan hasil
def storeResult(model, a,b,c,d):
    ML_Model.append(model)
    accuracy.append(round(a, 3))
    f1_score.append(round(b, 3))
    recall.append(round(c, 3))
    precision.append(round(d, 3))
    
    #Random Forest classifier model 
from sklearn.ensemble import GradientBoostingClassifier

#instantiate model
gr = GradientBoostingClassifier(max_depth=4,learning_rate=0.7)

#fit the model
gr.fit(X_train,y_train)

#predicting the target value from the model

y_train_gr = gr.predict(X_train)
y_test_gr = gr.predict(X_test)

#computting accuracy f1_store recall precision of the model performance

acc_train_gr = metrics.accuracy_score(y_train, y_train_gr)
acc_test_gr = metrics.accuracy_score(y_test, y_test_gr)
print("Gradien Bossting : Accuracy o Trainning Data :{:.3f}".format(acc_train_gr))
print("Gradien Bossting : Accuracy o Test Data :{:.3f}".format(acc_test_gr))
print()

f1_score_train_gr = metrics.f1_score(y_train, y_train_gr)
f1_score_test_gr = metrics.f1_score(y_test, y_test_gr)
print("Gradien Bossting : f1_score o Trainning Data :{:.3f}".format(f1_score_train_gr))
print("Gradien Bossting : f1_score o Test Data :{:.3f}".format(f1_score_test_gr))
print()

recall_score_train_gr = metrics.recall_score(y_train, y_train_gr)
recall_score_test_gr = metrics.recall_score(y_test, y_test_gr)
print("Gradien Bossting : Recall o Trainning Data :{:.3f}".format(recall_score_train_gr))
print("Gradien Bossting : Recall o Test Data :{:.3f}".format(recall_score_test_gr))
print()

precision_score_train_gr = metrics.precision_score(y_train, y_train_gr)
precision_score_test_gr = metrics.precision_score(y_test, y_test_gr)
print("Gradien Bossting : Precision o Trainning Data :{:.3f}".format(precision_score_train_gr))
print("Gradien Bossting : Precision o Test Data :{:.3f}".format(precision_score_test_gr))
print()

#computting classification of the model

print(metrics.classification_report(y_test,y_test_gr))

#storing the result. the bellow mentioned order to parameter passing important

storeResult('Gradient Boosting',acc_test_gr,f1_score_test_gr,recall_score_train_gr,precision_score_train_gr)

#creating Dataframe
result = pd.DataFrame({ 'ML Model' : ML_Model,
                        'Accuracy' : accuracy,
                        'f1_score' : f1_score,
                        'Recall'   : recall,
                        'Precision': precision,
                      })

#display total result
result

#sorting dataframe
sorted_result=result.sort_values(by=['Accuracy', 'f1_score'], ascending=False).reset_index(drop=True)

#menyimpan model terbaik.
from xgboost import XGBClassifier

gr = GradientBoostingClassifier(max_depth=4, learning_rate=0.7)

gr.fit(X_train, y_train)

#kita ubah jadi format pickle
import pickle

pickle.dump(gr, open('model_gradientboosting.pkl', 'wb'))

#check fitur importance in the model

plt.figure(figsize=(9,7))
n_features = X_train.shape[1]
plt.barh(range(n_features), gr.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_train.columns)
plt.title("Feature Importances using permutation on full model")
plt.xlabel("Feature Importance")
plt.ylabel("Freature")
plt.show()

#load model from file
loaded_model = pickle.load(open('model_gradientboosting.pkl', 'rb'))

# Melakukan prediksi dengan model yang telah dimuat
predictions = loaded_model.predict(X_test)