
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#accessing the dataset file 
mail = pd.read_csv('/content/mail_data.csv')

mail.head()

#getting info
mail.info()

#checking for null values
mail.isnull().sum()

mail.describe()

#replacing null values with empty strings
raw_mail = mail.where((pd.notnull(mail)),'')

#placing integer values for spam=0 ham=1
raw_mail.loc[raw_mail['Category']=='spam','Category',]=0
raw_mail.loc[raw_mail['Category']=='ham','Category',]=1

"""spam=0
ham=1

"""

X = raw_mail['Message']
Y = raw_mail['Category']

#printing first five rows of messages
X.head()

#printing first five rows of categories
Y.head()

#splitting the data into training and testing  
X_train,X_test,Y_train,Y_test = train_test_split( X, Y, test_size=0.2, random_state=3)

print(X.shape,X_train.shape,X_test.shape)

# transform the text data to feature vectors that can be used as input to the Logistic regression
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y_train and Y_test values as integers

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

print(X_train_features)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)

model.fit(X_train_features,Y_train)

#checking accuracy for train data
predictioin_trian=model.predict(X_train_features)
accuracy=accuracy_score(Y_train,predictioin_trian)
print('accuracy for trainging data is:',accuracy)

#checking accuracy for test data
prediction_test=model.predict(X_test_features)
accu=accuracy_score(Y_test,prediction_test)
print('accuracy for test data is',accu)

#testing our model
inp=[input('enter an email')]
input_data=feature_extraction.transform(inp)
prediction=model.predict(input_data)
if (prediction[0]==0):
  print('its spam mail')
else:
  print('its ham mail')

