# AI Fariness 360

The AI Fairness 360 Toolkit (AIF360) is a bias evaluation and mitigation framework for tabular data. It offers bias mitigation methods from the research knowledge sphere and bias evaluation functionality.

## Experiment

In my previous [research](../Additional%20Info/Investigating%20the%20impact%20of%20bias%20mitigation%20methods.pdf), we use the toolkit to benchmark its bias mitgation implementations against three datasets we gathered in Kaggle, [Job](https://www.kaggle.com/datasets/ayushtankha/70k-job-applicants-data-human-resource), [Bank](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data) and [College](https://www.kaggle.com/datasets/saddamazyazy/go-to-college-dataset). Those datasets are binary classification problems representing a real-life machine learning task with gender as the protected class. In addition, they are very varied in terms of their data size, gender representation and label distribution regarding gender. 

The bias mitigation implementation we accessed includes two pre-process methods, two in-process methods and two post-process methods:
    - [Reweighting](http://doi.org/10.1007/s10115-011-0463-8) and [DisparateImpactRemover](https://doi.org/10.1145/2783258.2783311)
    - [AdverservialDebiasing](https://arxiv.org/abs/1801.07593) and [GridSearchReduction](https://arxiv.org/abs/1905.12843)
    - [CalibratedEqOddsPostprocessing](https://papers.nips.cc/paper/7151-on-fairness-and-calibration) and [RejectOptionClassification](https://doi.org/10.1109/ICDM.2012.45)

## Result

The main finding of our experiment is that the fair-accuracy tradeoff depends on the training dataset. The datasets in our experiment are too varied, so we can not find substantial evidence on the performance of bias mitigation methods. In addition, we find that fairness metrics had a limited impact on the comparison of fairness-accuracy tradeoffs between methods.

One important thing is that in the case of the Bank dataset, where the training datasets have balanced gender representation and protected class correlation with positive labels, only in-processing methods can derive meaningful fairness improvements in terms of all bias metrics.

## Comment

### Reliability

- The installation of all dependencies of the framework requires the installation of R. However, without installing R, most of its features are still usable.
- Here are several bugs/problems I encountered during my prior research:
    - The conversion method from **Dataframe** to **StandardDataset** can break when given a **DataFrame** objects with 130 columns.
    - The bias mitigation implementation of **Reweight** and **RejectOptionClassifier** have problems when running.

### Generalizability

- Both bias detection and bias mitigation implementations only accept the toolkit's data structure **StandardDataset**
- However, the toolkit provides conversion methods to convert the **DataFrame** class to **StandardDataset**
- The conversion would replace categorical values with discrete integer values, remove rows with missing values, drop redundant features (specified in the parameter), identify favourable protected attributes and labels, and transform them to 0 and 1.
- This also means that the toolkit is limited to binary classification problems with binary privileged groups (privileged and protected)
- **StandardDataset** provides a method to split itself into training and testing sets randomly. It also provides a method to return a subset of itself based on a list of indexes.
- However, **StandardDataset** lack the flexibility and modifiability the **Dataframe** class provides. 
    - For example, you can split a **Dataframe** into five equal size folds and contact four folds to form the training sets for cross-validation. Such actions can not be performed with **StandardDataset** because it does not provide interfaces to combine different instances of **StandardDataset** together
- Bias metrics evaluation and bias mitigation implements can accept custom Machine Learning models to evaluate their bias metrics and mitigate its bias (except for in-process bias mitigation methods due to interfering with the training process directly)

### Guidance

- The README.md of its GitHub repository provides detailed information on its instalments. It also provides several instalment options with different supported features and dependencies
- There are plentiful learning resources in its GitHub repository for new users to get familiar with the toolkit.
- The learning resources include:
    - A demo notebook with each bias mitigation method
    - Several tutorials on using the toolkits
- Its methods and interfaces are well documented for the parameter, and it has provided detailed [documentation](https://aif360.readthedocs.io/en/stable/index.html)
- It has provided mathematical definitions of each supported bias metric on the documentation website
- It also provides example datasets for new users to experiment with

### Robustness

- The framework uses the **StandardDataset** class as its data structure provides convenience, such as automating data pre-process tasks, transforming categorical features, and categorising privileged groups. 
    - However, it is less flexible than Panda's **Dataframe** class to query its content and perform various transformation operations, making it harder to debug if problems are encountered when using the framework.
    - Encountering problems during conversion between **Dataframe** and **StandardDataset** means users can not perform bias evaluation and mitigation with this framework
    - In addition, **StandardDataset** do not use most of the **DataFrame**'s interface, meaning when a project involves using both **Dataframe** and **StandardDataset**, the user has to prepare different methods to account for both classes (i.e. retrieve index of a dataset or execute a cross-validation split)
- It provides evaluations of plentiful bias metrics. However, most of them have definitions based on prediction and ground truth or predictions themselves, lacking those definitions based on predicted outcome, similarity measures and causal reasoning 
- It provides implementations of state-of-art bias mitigation methods that can be very convenient for any machine learning workflow to improve their fairness. However, there are significantly more implementations of in-processing mitigation methods than pre-processing and post-processing ones.
    - Since implementations of in-processing mitigation methods present a new machine learning training process rather than modifying existing ones, they can not be able to integrate with existing workflows and have to actively compete with them

## How to utilise this subrepo

This subrepo has been used in Windows environments. 

To use this subrepo in the Window Environment, use Python version 3.11 and [window_requirements.txt](./window_requirements.txt) with pip to install the virtual environment.

This subrepo only contains the benchmark experiment part of our previous research; my partner handles data analysis and visualisation.

*Bank*, *College* and *Job* contain the Jupyter notebook conducting the experiment and CSV files extracted from them. The CSV files record each epoch metrics result from each model in the dataset. The code in the notebook is modified and improved as *experiment_util.py* and *modesl.py* custom modules used in ***Aequitas*** and ***Fairness Indicators*** subrepos.