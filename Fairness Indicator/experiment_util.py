import tensorflow_model_analysis as tfma
import tensorflow_data_validation as tfdv
from google.protobuf import text_format
import os
import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset
import shutil
#Global values
OUTPUT_PATH=os.path.abspath("bin")
with open("eval.config","r") as file:
    config=file.read()
column=["Accuracy Difference",	"FPR Difference", "FNR Difference","TPR Difference","TNR Difference","PR Difference",
         "NR Difference",	"FDR Difference", "FOR Difference", "Precision Difference","Recall Difference"]

#normalizing dataframe numeric value
def min_max_noramlize(column):
    return (column-column.min())/(column.max()-column.min())

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
        eval_config = text_format.Parse(config, tfma.EvalConfig())
        try:
            shutil.rmtree(OUTPUT_PATH)
        except OSError:
            pass
        eval_result = tfma.analyze_raw_data(data=result,eval_config=eval_config,output_path=OUTPUT_PATH)
        metrics_difference=self.extract_result(eval_result)
        accuracy=pd.Series([get_accuracy(result,"prediction","label")],index=["Accuracy"])
        return pd.concat([accuracy,metrics_difference])

    #Extract values from Fairness indcator's result
    def extract_result(self, eval_result):
        for sn in eval_result.get_slice_names():
            row=[]
            for key,item in eval_result.get_metrics_for_slice(sn).items():
                if key=="example_count":
                    continue
                data=item["doubleValue"]
                if data=="NaN":
                    row.append(np.nan)
                else:
                    row.append(data)
            if self.pg in str(sn):
                male_row=pd.Series(row,index=column)
            else:
                female_row=pd.Series(row,index=column)
        return male_row-female_row