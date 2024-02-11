import os
import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset
import shutil
#Global values

column=["Accuracy Difference",	"FPR Difference", "FNR Difference","TPR Difference","TNR Difference","PR Difference",
         "NR Difference",	"FDR Difference", "FOR Difference", "Precision Difference","Recall Difference"]

#normalizing dataframe numeric value
def min_max_noramlize(column):
    return (column-column.min())/(column.max()-column.min())

def preporcess_transform(df):
    numeric_feautre_names=df.select_dtypes(include=["number"]).columns
    categorical_feature_names=df.select_dtypes(include=["object_"]).columns

    pre_df=df.copy()

    for nc in numeric_feautre_names:
        pre_df[nc]=min_max_noramlize(pre_df[nc])   

    for cc in categorical_feature_names:
        pre_df[cc]=pd.factorize(pre_df[cc])[0]
    return pre_df

#append a Series to an exisitng Dataframe
def append_Series(dataframe, series):
    df_name=dataframe.columns.name
    new_df=pd.concat([dataframe,pd.DataFrame(series).T])
    new_df.columns.name=df_name
    return new_df

def get_accuracy(dataframe, pred_column, label_column):
    return dataframe[dataframe[pred_column]==dataframe[label_column]].shape[0]/dataframe.shape[0]

# Helper class for handling evaluation emthods for certain datasets
class evaluation:
    def __init__(self,df,gender_column,label_column,privileged_group,split_fold) -> None:
        self.df=df
        self.gc=gender_column
        self.lc=label_column
        self.folds=split_fold
        self.pg=privileged_group
    
    def get_train_test(self,dataset,fold_index):
        test_fold=self.folds[fold_index]
        train_index=pd.Index([])
        for f in self.folds:
            if f is not test_fold:
                train_index=train_index.append(f)
        if type(dataset) is pd.DataFrame:
            train_set=dataset.loc[train_index]
            test_set=dataset.loc[test_fold]
        if type(dataset) is StandardDataset:
            train_set=dataset.subset(train_index)
            test_set=dataset.subset(test_fold)
        return train_set,test_set

    # Cross vailidation and return the average result
    def cross_validation(self,row_name,train_test_process,dataset):
        result=[]
        for i in range(5):
            train_set, test_set=self.get_train_test(dataset,i)
            ground_truth,test_pred=train_test_process(train_set,test_set,self)
            epoch_result=self.framework_evaluate(ground_truth,test_pred)
            result.append(epoch_result)
        cv_result=sum(result)/5
        cv_result.name=row_name
        return cv_result
        
    
    #Evaluate with Fairness indicator
    def framework_evaluate(self,ground_truth,prediction):
        pred=pd.Series(prediction,index=ground_truth.index,name="pred")
        result=pd.concat([self.df.loc[ground_truth.index,[self.gc]],ground_truth,pred],axis=1)
        result.columns=["gender","label","prediction"]
        pass