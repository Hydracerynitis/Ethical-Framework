import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from aequitas.flow.methods.inprocessing.fairgbm import FairGBM
from aequitas.flow.methods.inprocessing.fairlearn_classifier import FairlearnClassifier
from fairlearn.reductions import ExponentiatedGradient, FalsePositiveRateParity
from lightgbm import LGBMClassifier

def get_feautures(dataset,label_column):
    feature=dataset.loc[:,dataset.columns!=label_column]
    label=dataset[label_column]
    return feature,label

def RandomForest(train_set, test_set,eval):
    #Training
    model=RandomForestClassifier()
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

def Fairgbm(train_set,test_set,eval):
    #Training
    model=FairGBM(
        global_constraint_type="FPR,FNR",
        global_target_fpr=0.05,
        global_target_fnr=0.5,
        constraint_type='fpr',
        multiplier_learning_rate=0.1,
        constraint_stepwise_proxy="cross_entropy",
        boosting_type="dart",
        enable_bundle=False,
        num_leaves=10,
        n_estimators=100,
        min_child_samples=50,
        learning_rate=0.01
    )
    train_X,train_Y=get_feautures(train_set,eval.lc)
    model.fit(train_X,train_Y,train_X[eval.gc])
    #Testing
    test_X,test_Y=get_feautures(test_set,eval.lc)
    pred=model.predict_proba(test_X,test_X[eval.gc])
    return test_Y, pred

def Fairlearn(train_set,test_set,eval):
    model=FairlearnClassifier(ExponentiatedGradient,LGBMClassifier,FalsePositiveRateParity,
        eps=0.05,
        max_iter=10,
        model__boosting_type="dart",
        model__enable_bundle=False,
        model__n_estimators=100,
        model__num_leaves=10,
        model__min_child_samples=50,
        model__learning_rate=0.01,
        model__n_jobs=1
    )
    train_X,train_Y=get_feautures(train_set,eval.lc)
    model.fit(train_X,train_Y,train_X[eval.gc])
    #Testing
    test_X,test_Y=get_feautures(test_set,eval.lc)
    pred=model.predict_proba(test_X,test_X[eval.gc])
    return test_Y, pred