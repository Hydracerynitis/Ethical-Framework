import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing.disparate_impact_remover import DisparateImpactRemover

def get_feautures(dataset,label_column):
    if type(dataset) is pd.DataFrame:
        feature=dataset.loc[:,dataset.columns!=label_column]
        label=dataset[label_column]
    if type(dataset) is StandardDataset:
        feature=dataset.features
        label=dataset.labels.ravel()
    return feature,label

def get_index(dataset):
    if type(dataset) is StandardDataset:
        return [int(i) for i in dataset.instance_names]
    if type(dataset) is pd.DataFrame:
        return dataset.index

def RandomForest(train_set, test_set,eval):
    #Training
    model=RandomForestClassifier()
    train_X,train_Y=get_feautures(train_set,eval.lc)
    model.fit(train_X,train_Y)
    #Testing
    test_X,test_Y=get_feautures(test_set,eval.lc)
    pred=model.predict(test_X)
    return test_Y, pred

def NeuralNetwork(train_set,test_set,eval):
    #Training
    model=MLPClassifier(
        hidden_layer_sizes= 20,
        solver="sgd",
        learning_rate_init=0.01,
        early_stopping=False,
        alpha=1
    )
    train_X,train_Y=get_feautures(train_set,eval.lc)
    model.fit(train_X,train_Y)
    #Testing
    test_X,test_Y=get_feautures(test_set,eval.lc)
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

def DI_RandomForest(train_set,test_set,eval):
    #Training
    di = DisparateImpactRemover(repair_level=0.75,sensitive_attribute=eval.gc)
    train_di = di.fit_transform(train_set.copy(deepcopy=True))
    test_di = di.fit_transform(test_set.copy(deepcopy=True))

    di_model=RandomForestClassifier()
    di_model.fit(train_di.features,train_di.labels.ravel(),sample_weight=train_di.instance_weights)
    #Testing
    pred=di_model.predict(test_di.features)
    return pd.Series(test_set.labels.ravel(),index=get_index(test_set)),pred

def DI_NeuralNetwork(train_set,test_set,eval):
    #Training
    di = DisparateImpactRemover(repair_level=0.75,sensitive_attribute=eval.gc)
    train_di = di.fit_transform(train_set.copy(deepcopy=True))
    test_di = di.fit_transform(test_set.copy(deepcopy=True))

    di_model=MLPClassifier(
        hidden_layer_sizes= 20,
        solver="sgd",
        learning_rate_init=0.01,
        early_stopping=False,
        alpha=1
    )
    di_model.fit(train_di.features,train_di.labels.ravel())
    #Testing
    pred=di_model.predict(test_di.features)
    return pd.Series(test_set.labels.ravel(),index=get_index(test_set)),pred

def DI_GradientBoost(train_set,test_set,eval):
    #Training
    di = DisparateImpactRemover(repair_level=0.75,sensitive_attribute=eval.gc)
    train_di = di.fit_transform(train_set.copy(deepcopy=True))
    test_di = di.fit_transform(test_set.copy(deepcopy=True))

    di_model=GradientBoostingClassifier()
    di_model.fit(train_di.features,train_di.labels.ravel())
    #Testing
    pred=di_model.predict(test_di.features)
    return pd.Series(test_set.labels.ravel(),index=get_index(test_set)),pred