import numpy as np
import pandas as pd
import re as re

train = pd.read_csv('./data_input/train.csv', header = 0, dtype={'Age': np.float64})
test = pd.read_csv('./data_input/test.csv'  , header = 0, dtype={'Age': np.float64})
full_data = [train, test]

##print (train.info())

##print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())

##print (train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())

for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

#print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())

for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

##print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
##print (train[['FamilySize', 'Survived']])

for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    
##print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())

for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
#print (train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())


for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)
##print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

#print(pd.crosstab(train['Title'], train['Sex']))


for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countness','Capt','Col', \
                       'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

