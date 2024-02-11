# Aequitas

Aequitas is an open-source bias auditing and Fair Machine Learning toolkit. The objective of this package is to provide an easy-to-use and transparent tool for auditing machine learning models, allowing user to correct model with bias mitigation methods, as well as an automated benchmark system for systematic evaluation.

## Experiment


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

- User can use *yaml* files to config an experiment 

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
