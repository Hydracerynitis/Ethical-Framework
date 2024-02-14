# Aequitas

Aequitas is an open-source bias auditing and Fair Machine Learning toolkit. This package aims to provide an easy-to-use and transparent tool for auditing machine learning models, allowing users to correct models with bias mitigation methods and an automated benchmark system to find the models with the best performance-fairness tradeoff.

## Experiment

In the experiment I set up with Aequitas, I want to investigate the effectiveness of Aequitas's bias mitigation implementations. I will test Aequitas's **FairGBM** and **FairLearnClassifier** against [Diabetes](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), [OULAD](https://analyse.kmi.open.ac.uk/open_dataset) and [College](https://www.kaggle.com/datasets/saddamazyazy/go-to-college-dataset), which all have balanced label distributions and balanced gender representation.  

Throughout this experiment, I only used one set of hyperparameters for **FairGBM** and **FairLearnClassifier**. The hyperparameters are set according to the restrictions in the framework example's [config](./demo/examples/configs/paper_configs/methods/fairgbm_baf/inprocessing/fairgbm.yaml) [files](./demo/examples/configs/paper_configs/methods/exponentiated_gradient_baf/inprocessing/exponentiated_gradient.yaml) to replicate the performance seen in the framework's paper. The exact set of hyperparameters can be seen in **[models.py](./models.py)** files.

I use bias metrics of False Omission Rate Disparity, False Discovery Rate Disparity, Predicted Positive Ratio Disparity, and Predicted Prevalence Disparity to test against both implementations. False Omission Rate Disparity and False Discovery Rate Disparity examine models' fairness in light of equality. On the other hand, predicted Positive Ratio Disparity and Predicted Prevalence Disparity examine models' fairness in light of equity. The reference group among the metrics is Males, the privileged group. 

In the experiment, **FairGBM** and **FairLearnClassifier** will be accessed along with scikit-learn's **RandomForestClassifier** and **GradientBoostingClassifier** as comparison groups. The confidence level throughout the experiment will be 95%. Since **FairGBM** and **FairLearnClassifier**'s predictions are probabilistic rather than binary, their prediction will be transformed into binary based on a 0.5 threshold.

## Experiment Result

In the Diabetes dataset, **FairGBM** and **FairLearnClassifier** provide limited improvement across all bias metrics while trading noticeable accuracy compared to the comparison group. Both models have the same significance test results as the comparison group, except on Predicted Prevalence Disparity, where only **FairGBM** pass the significance test.

In the OULAD dataset, **FairGBM** and **FairLearnClassifier** perform better on Predicted Positive Ratio Disparity and Predicted Prevalence Disparity, making them pass the significance test on Predicted Prevalence Disparity, while the comparison group failed. However, they have similar False Discovery Rate Disparity and even worse False Omission Rate Disparity than the comparison group. Performance-wise, **FairGBM** and **FairLearnClassifier** remain between **RandomForestClassifier** and **GradientBoostingClassifier**, with **GradientBoostingClassifier** leading in accuracy.

The result is much more drastic in the College dataset, which may be due to the college dataset being a synthetic dataset with low sample sizes. **FairGBM** and **FairLearnClassifier** provide limited and sometimes worse bias metrics than the comparison group while trading away accuracy. The only exception is False Discovery Rate Disparity, where **FairLearnClassifier** overcorrects significantly while **FairGBM** improves the bias metrics.

Overall, the effectiveness of **FairGBM** and **FairLearnClassifier** is unclear from the limited observation in this experiment. It is assumed that there are other characteristics among those datasets influence the effectiveness of **FairGBM** and **FairLearnClassifier**, which are worth investigating.

## Comment

### Reliability

- Aequitas's calculation of confusion matrix metrics disparity is hard-coded to require input datasets to have "label_value" in their column to proceed. It does not offer any parameter for users to specify the names of the dataset's label column.
    - This also violates the assumption set up by its previous step, where you can specify the names of the dataset's label column for it to proceed
    - The default label column name of the previous step is "label", which contradicts the following steps' requirement.
- Due to how Aequitas imports bias mitigation modules, you need to download and compile [**LightGBM.dll**](https://github.com/microsoft/LightGBM/releases) Libary and put it in the virtual environment's library folders. There is no mention of it in its installation information.
    - Likewise, its automated benchmark feature also requires dependency on either pyarrow or fastparquet for parquet support, which is missing from the installation information.
    - Despite that, you need to install **lightgbm** and **torch** (preferably with CUDA support) to use the lightgbm provided as its base estimator. These are not included in its dependency and will not be installed with **pip install aequitas** command
- When reading the result of an automated benchmark experiment, Aequitas will assume it's reading from the Linux system and does not account for the Windows system. As a result, it will throw an error when analysing analyseult of an automated benchmark run, making it unusable. 
- When replicating the tutorial's codes, I found that the postprocessing methods **BalancedGroupThreshold** change nothing after executions. The causes of this problem are unclear.

### Generalizability

- To produce a report of models' bias metric, it only needs its prediction and the corresponding ground truth and protected attribute values. The user can specify a reference group when calculating the disparity of fairness metrics with other groups
- Users have great control over Aequitas's automated benchmark system. They can specify almost every benchmark detail by using the *yaml* format config file. The details of the benchmark experiment that users can specify include:
    - The methods class that will be used in the experiment and the range for its hyperparameters
        - It may mean for users to use custom model classes in the experiment, they should give it the same interface as aequitas's **LightGBM** class.
    - The preprocessing, inprocessing and postprocessing methods that are used to train models
    - The dataset class that will be used in the experiment and its hyperparameters
        - It may mean that you must write a custom dataset code handling read data from a source and provide similiar interfaces like aequitas's **BankAccountFraud** class.
    - Numbers of runs performed in the experiment and the sampling methods to sample hyperparameters to train different models.

### Guidance

- Aequitas provide a notebook for each tutorial for its auditing model's predictions, correcting the model's prediction and training the fairness-aware model feature. However, the tutorials will suddenly use variable names that it has not created before, nor have it referred to, making it confusing to follow. 
    - Such a problem is observed in [auditing model's predictions](https://colab.research.google.com/github/dssg/aequitas/blob/notebooks/compas_demo.ipynb) tutorial
- All tutorials seem not to concern themselves with providing reasoning behind their choice of metrics to determine whether the model is fair. They assume users already have some basic understanding of the usages and tradeoffs of provided fairness metrics.
- Aequitas Github Pages and its tutorial provide definitions of absolute and disparity metrics. However, it lacks definitions for its fairness metrics.
    - Since aequitas only provides True or False values for fairness metrics, indicating whether the model's fairness metrics have crossed the thresholds, their definition are more needed to understand the meaning they represent fully and the threshold value.
 
### Robustness

- Aequitas offers large ranges of absolute, disparity, and fairness metrics. As a result, it will cover most demands of a fairness evaluation
- Conversely, it may be too overwhelming for users to understand the result and design further processing. As a result, it is more suited for users who
    - Fully understand all confusion matrix metrics and what they represent in real-world practices
    - Have a clear view of fairness they demand for their models to have; thus have already decided which metrics to use in their evaluation.
- Most absolute bias metrics aequitas provides have definitions based on predictions and ground truth, while only two bias metrics have definitions based on predictions. This means that aequitas leans more toward equality as fairness rather than equity as fairness.
    - It means aequitas is less relevant when the demands for fairness focus on equity
- The audition feature is built for auditing a single prediction of a single model. It cannot also calculate metrics of the model's overall performance. Thus, additional effort is required to process cross-validation results and multiple model comparison
    - However, Aequitas use **DataFrame** as its metrics output, meaning its data can be easily extracted and transformed due to how flexible **DataFrame** is.
- Aequitas's bias mitigation implementations, more specifically **FairGBM**, are not compatible with the Windows environment.
    - Also, when Aequitas extracts results from a benchmark run's artifacts, it does not account for how the directory will be expressed in the Windows environment. As a result, the automated benchmark system is also incompatible with the Windows environment.
- Aequitas's automated benchmark system is impressive. However, its visualisation shows the hyperparameters used when training the models. It may mean that it is impossible to replicate the recommended models shown in the benchmark result. 

## How to utilise this subrepo

This subrepo has been used in both Windows and Linux environments. 

This subrepo will utilise CUDA. My Windows CUDA driver version is 12.3, and my Linux CUDA driver version is 11.6

To use this subrepo in the Window Environment, use Python version 3.11 and [window_requirements.txt](./window_requirements.txt) with pip to install the virtual environment. **Be mindful that Aequitas's bias mitigation implementations and automated benchmark system features are not compatible** 

To use this subrepo in Linux Environment, use Python version 3.10 and [linux_requirements.txt](./linux_requirements.txt) with pip to install the virtual environment.

*College_experiment.ipynb*, *diabetes_experiment.ipynb*, and *OULAD_experiment*.ipynb are notebooks I used to construct the mockup experiment. They use my custom modules in the same folders: *experiment_util.py*, *models.py*, and *plot_util.py*

- *experiment_util.py* provides global variables that can be used in the experiment and methods to transform the **DataFrame** object. It also offers the **Evaluation** class, which stores information about the datasets, the name of its column containing label value and sensitive attributes, the symbol representing the privileged groups, and separate sets of indexes for cross-validation.
    - Its methods *get_train_test* take a preprocessed dataset and the index of the set chosen for test fold and output training dataset and testing dataset retrieved from the preprocessed dataset
    - Its methods *framework_evaluate* take Series containing ground truth and models' prediction and output Series containing Accuracy metrics and bias metrics retrieved from aequitas
    - Its methods *cross_validation* take names of the models, training and testing methods from *modules.py*, and preprocessed datasets, use *get_train_test* to get training and testing sets for each epoch of cross-validation and feed them to the training and testing methods, then feed prediction and ground truth of each epoch to *framework_evaluate* to retrieve the epoch's metrics. In the end, it calculates the average of each metric and outputs them in a Series
- *models.py* provides training and testing methods for each accessed model algorithm
- *plot_util* provides a method to draw a scatter graph to visualise result

*demo* folders contain notebooks I used to follow the framework's tutorial to understand the framework and custom modules I copied from other experiments to reduce the effort when learning. The *examples* folder contains config files that the author of the framework used in their paper and their tutorials. They can also be used in your benchmark experiment with Aequitas.

