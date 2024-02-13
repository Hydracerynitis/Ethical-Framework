import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

def get_feautures(dataset,label_column):
    feature=dataset.loc[:,dataset.columns!=label_column]
    label=dataset[label_column]
    return feature,label

def RandomForest(train_set, test_set,label_column):
    #Training
    model=RandomForestClassifier()
    train_X,train_Y=get_feautures(train_set,label_column)
    model.fit(train_X,train_Y)
    #Testing
    test_X,test_Y=get_feautures(test_set,label_column)
    pred=model.predict(test_X)
    return test_Y, pred


def GradientBoost(train_set,test_set,eval):
    #Training
    model=GradientBoostingClassifier()
    train_X,train_Y=get_feautures(train_set,eval.lc)
    model.fit(train_X,train_Y)
    #Testing
    test_X,test_Y=get_feautures(test_set,eval.lc)
    pred=model.predict(test_X)
    return test_Y, pred