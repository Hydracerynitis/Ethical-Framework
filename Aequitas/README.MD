# Aequitas

Aequitas is an open-source bias auditing and Fair Machine Learning toolkit. The objective of this package is to provide an easy-to-use and transparent tool for auditing machine learning models, allowing user to correct model with bias mitigation methods, as well as an automated benchmark system to find the models with the best performance-fairness tradeoff.

## Experiment

In the experiment I set up with Aequitas, I want to investigate the effectiveness of Aequitas's bias mitigation implementations. I will test Aequitas's **FairGBM** and **FairLearnClassifier** against [Diabetes](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), [OULAD](https://analyse.kmi.open.ac.uk/open_dataset) and [College](https://www.kaggle.com/datasets/saddamazyazy/go-to-college-dataset), which all have balanced label distributions and balanced gender representation.  

For this experiment, I only use one sets of parameters for **FairGBM** and **FairLearnClassifier**. The parameters are followed by the restrictions set in the framework example's [config](./demo/examples/configs/paper_configs/methods/fairgbm_baf/inprocessing/fairgbm.yaml) [files](./demo/examples/configs/paper_configs/methods/exponentiated_gradient_baf/inprocessing/exponentiated_gradient.yaml) to replicate the performance seen in the framework's paper. The exact set of parameters can be seen in **[models.py](./models.py)** files.

I use bias metrics of False Omission Rate Disparity, False Discovery Rate Disparity, Predicted Positive Ratio Disparity, and Predicted Prevalance Disparity to test against both implementations. False Omission Rate Disparity and False Discovery Rate Disparity examine models' fairness in light of equality. Predicted Positive Ratio Disparity and Predicted Prevalance Disparity on the other hand examine models' fairness in light of equity. The refernce group among the metrics is Male, the privilleged group. 

In the experiment, **FairGBM** and **FairLearnClassifier** will be accessed along with scikit-learn's **RandomForestClassifier** and **GradientBoostingClassifier** as comparison groups. The confidence level throughout the experiment will be 95%. Since **FairGBM** and **FairLearnClassifier**'s predictions are probabilisitic rather than binary, their prediction will be transformed into binary based on 0.5 threshold.

## Experiment Result

In Diabetes dataset, **FairGBM** and **FairLearnClassifier** provides limited improvement accross all bias metrics, while trading noticable accuracy compared to the comparison group. **FairGBM** seems to work best on False Omission Rate Disparity and Predicted Prevalance Disparity, making it the only model that pass the significance test on both metrics. 

In OULAD dataset, **FairGBM** and **FairLearnClassifier** perform better on False Omission Rate Disparity,  Predicted Positive Ratio Disparity, and Predicted Prevalance Disparity, making them pass the significance test on False Omission Rate Disparity and Predicted Prevalance Disparity while the comparison group fail. However, they have worse False Discovery Rate Disparity than the comparison group.
Performance wise, **FairGBM** and **FairLearnClassifier** remain between **RandomForestClassifier** and **GradientBoostingClassifier**, with **GradientBoostingClassifier** leads in accuravy.

In College dataset, the result is much more varied. This may be attributed to the college dataset being a synthetic dataset as well as having low sample sizes. **FairGBM** provide a significant improvement on False Omission Rate Disparity


## Comment

### Reliability

- Framework's calculation of confusion matrix metrics disparity is hard-coded to require input datasets to have "label_value" in its column to proceed. It does not offer parameter for users to specify the names of the dataset's label column.
    - This also violate assumption set up by its previous step, where you can specify the names of the dataset's label column for it to proceed
    - The default label column name of previous step is "label", which contradict with the next steps' requirement.
- Due to how Aequitas import bias mitigation modules, you need to download and compile [**LightGBM.dll**](https://github.com/microsoft/LightGBM/releases) Libary and put it in the virtual environment's library folders. There is no mention of it in its installation information.
    - Likewise, its automated benchmark feature also require dependency of either pyarrow or fastparquet for parquet support, which is missing from the instalation information.
    - Despite of that, you still need to install **lightgbm** and **torch** (preferably with cuda support) yourself to use the lightgbm provided as its base estimator. These are not included in its dependency and will not be installed with **pip install aequitas** command
- When reading the result of an automated benchmark experiment, Aequitas will assume it's reading from Linux system and do not account for Window system. As a result, it will throw an error when try to anlysis the result of an automated benchmark run, making it unusable. 
- When replicating tutorial's codes, I found that the post-processing methods **BalancedGroupThreshold** change nothing after executions. It is unclear to me what is the causes of this problem.

### Generalizability

- To produce a report of models' bias metric, it only needs its prediction and the correspounding ground truth and protected attribute values. The user can specify a reference group when calculating the disparity of fairness metrics with other groups
- User can use *yaml* files to plan a benchmark 

### Guidance

- Aequitas provide notebook each tutorial for its auditing model's predictions, correcting model's prediction and training fairness aware model feature. However, the tutorials will suddenly use variable names that it has not created before nor it has it referred, making it confusing to follow. 
    - Such probelm is observed in [auditing model's predictions](https://colab.research.google.com/github/dssg/aequitas/blob/notebooks/compas_demo.ipynb) tutorial
- All tutorial seem to not concern themselves to provide reasoning behind their choice of metrics to determine whether the model is fair. They assume users have already some basic understanding of the usages and tradeoff of provided fairness metrics.
- Aequitas Github Pages and its tutorial provide definitions of abosulte metrics, disparity metrics. However, it lack the definition for its fairness metrics.
    - Since aequitas only provide True or False value for fairness metrics, indicating whether model's fairness metrics has cross the thresholds, their definition are more needed to fully understand the meaning they represent as well as the threshold value.
 
### Robustness

- Aequitas offers large ranges of absolute metrics, disparity metrics and fairness metrics. Without any filtering or selection,
- On the other hand, it may be too overwhelming for users to understand the result and desgin furhter processing. As a result, it is more suited for users who
    - Fully understand all confusion matrix metrics and what they represent in real world pratices
    - Have a clear view of fairness they demand for their models to have, thus have already decided which metrics to use in their evaluation.
- Aequitas only provides **DataFrame** as their metrics output, meaning there is no convenient way to extract a specific metrics of a specify groups for further processing.
