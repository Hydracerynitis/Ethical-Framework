# Fairness Indicator

Fairness Indicator is a bias evaluation framework built upon Tensorflow Model Analysis. It positions itself for the TensorFlow machine learning model pipeline while also packaging with the What-If widget.

## Experiment

I set up the experiment with a fairness indicator to investigate the effectiveness of [Disparate Impact Remover](https://doi.org/10.1145/2783258.2783311) (DIR) by AI Fairness 360. I will compare the difference between the accuracy-bias-metrics-tradeoff of fairness-unaware models and that of the same models with Disparate Impact Remover 

The fairness metrics in the experiments will be offered by the fairness indicator, which includes false positive rate, false negative rate, false discovery rate and false omission rate. To evaluate the performance of Disparate Impact Remover, I will compare the change in those bias metrics difference between protected groups and privileged groups.

The models I accessed in the experiment are random forest, multilayer neural networks, and gradient boosting models. For the hyperparameters of Disparate Impact Remover, I only used 0.75 repair levels due to the survey's timeframe.

The datasets I used in the experiment are [Adult](https://archive.ics.uci.edu/dataset/2/adult), [OULAD](https://analyse.kmi.open.ac.uk/open_dataset), [Job](https://www.kaggle.com/datasets/ayushtankha/70k-job-applicants-data-human-resource), and [Bank](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data) datasets. The selection of datasets is to investigate the effectiveness of Disparate Impact Remover in different cases of gender representation and gender distribution.

The characteristics of individual datasets are the following: 
- Adult dataset has an imbalanced label distribution and an imbalanced gender representation. 
- OULAD dataset has a balanced label distribution and a balanced gender representation. 
- Job dataset has a balanced label distribution and an imbalanced gender representation.
- Bank dataset has an imbalanced label distribution and a balanced gender representation. 

## Experiment Result

Applying the Disparate Impact Remover to the Multilayer Neutral Network training process will result in the model only predicting negatives. This problem occurs across every dataset. Its false positive rate difference, false negative rate difference, false discovery rate difference and false omission rate difference become invalid and unfit for comparison. It is unclear what caused this issue. As a result, the Neural Network model is excluded from the analysis. 

Disparate Impact Remover makes a slight difference in all models across all datasets. However, Disparate Impact Remover's effect on bias metrics varies depending on the training datasets. In the Adult dataset, it does help improve models' False Positive Rate difference and False Negative Rate difference. However, it worsens the Random Forest model's False Discovery Rate difference and False Omission Rate difference while making almost no change to the Gradient Boosting model. In the case of the OULAD and Job datasets, it either overcorrects the models' bias metrics difference or worsens them. In the case of the Bank dataset, it either makes slight improvements or almost no change in bias metrics difference.

Overall, the Disparate Impact Remover seems to work best when the training dataset has an imbalanced label distribution. Having a balanced gender representation allows it to improve the model's fairness more effectively. Disparate Impact Remover can improve the model's False Positive Rate Difference and False Negative Rate Difference. However, its effectiveness regarding False Discovery Rate Difference and False Omission Rate Difference is limited.

One thing that must be pointed out is that Disparate Impact Remover is intended to make a model make a similar amount of positive or negative predictions regardless of whether the individual belongs to privileged groups or protected groups. By its definition, it is unfit to improve the model's overall correctness regarding privileged groups and protected groups. As a result, the experiment's conclusion is limited and can even be considered misleading.

## Comment

### Reliability

- The Fairness Indicator is built upon TensorFlow Model Analysis, which is only available in Python 3.8 and 3.9. This installation prerequisite is not mentioned on either the GitHub page or tutorials.
- There are several instances of dependency conflict when installing fairness indicators both on Windows and Linux:
    - Installing the **fairness-indicators** of version > 0.40.0 on a newly created virtual environment (Python 3.8 and 3.9) will conflict with its dependency **apache-beam**. This issue makes them uninstallable through pip.
        - This is due to **fairness-indicators**'s dependency **tensorflow-model-analysis** requires package **attrs** of version >=19.3.0 and <0.22, while **apache-beam**'s dependency **jsonschema** requires package **attrs** of version >=22.2.0
    - Installing **fairness-indicators** and **scikit-learn** (as well as **aif360** which depends on **scikit-learn**) will cause a dependency conflict
        - This is due to **fairness-indicators**'s dependency **tensorflow-data-validation** requires package **joblib** of version <0.15 and >=0.12 while **scikit-learn** (version 1.3.2) requires package **joblib** of version 1.3.2
        - This can be resolved by backtracking to the older version of **scikit-learn** (<=1.0.2)
    - To correctly install the framework, you need to install its dependency first and then install the framework itself. This information can be found on the framework's [github page](https://github.com/tensorflow/fairness-indicators)
- Throughout the experiment, I have encountered various problems that prevent me from using AI Fairness 360 implementations. However, whether this is due to older version Python packages or bugs in the AIF360 implementations is unsure.
    - Affected implementations includes **AdversialDebiasing**, **GridSearchReduction**
- The framework will output the string "NaN" when calculating the metrics when the model only makes positive predictions or negative predictions. It will become an error if the evaluation pipeline does not take this into account.

### Generalizability

- A small thing to note is that the framework's visualisation assumes the IDE theme users are using has a white background. This has forced me to switch my IDE theme from a black background to a white background to be able to see the results.
- To produce a report of models' bias metric, it only needs its prediction and the corresponding ground truth and protected attribute values. The user can use a config file (or write configuration in code) to specify the columns of **DataFrame** to retrieve protected attributes, ground truth, models' predictions, and the metrics it will calculate from them.


### Guidance

- It provides Colab notebook tutorials to guide new users. However, the notebook is run on Python version 3.10. As Fairness-Indicators' prerequisite is Python version 3.8 and 3.9, those notebook tutorials are not working.
    - I have encountered a problem with the framework that can be attributed to dependency version conflict, where the framework exhibits different behaviour that results in an error.

### Robustness

- One thing to note is that I only evaluate the framework's performance regarding non-tensorflow models since I could not learn and fully understand how to create and train a tensorflow model during the survey's timeframe. 
    - Since the framework is built on the TensorFlow library, I assume it could provide a more in-depth and better analysis of the TensorFlow model than the non-tensorflow model and better integration into the TensorFlow model workflow.
- The Fairness Indicator framework can be easily integrated into the model evaluation phase of the machine learning workflow and customised to cater to users' requirements.
- However, The framework is built for comparison between the performance of privileged groups and protected groups from a single prediction of a single model. It cannot also calculate metrics of the model's overall performance
    - It's complicated to extract data from the framework for further analysis; extra methods are required for the user for process cross-validation results and multiple model comparison
- The definitions of available metrics consist of only those based on predicted outcomes and ground truth. The framework also presents its data in the form of comparisons between privileged groups and protected groups.
    - This makes the framework much weaker when ensuring the model achieves equal accuracy or other metrics is not ideal or even impossible. (i.e. Disparate Impact)

## How to utilise this subrepo

This subrepo has been used in Windows environments. 

To use this subrepo in the Window Environment, use Python version 3.8 and [window_requirements.txt](./window_requirements.txt) with pip to install the virtual environment. Alternatively, install tensorflow==2.9, tensorflow-data-validation==1.9.0, tensorflow-model-analysis=0.40.0, scikit-learn==1.0.2, and then install fairness-indicators==0.40.0

*demo.ipynb* is used to learn using Fairness Indicator with Dataframe as its input in its [tutorial](https://colab.research.google.com/github/tensorflow/fairness-indicators/blob/master/g3doc/tutorials/Fairness_Indicators_Pandas_Case_Study.ipynb). 

*Adult Experiment.ipynb*, *Bank Experiment.ipynb*, *Job Experiment.ipynb* and *OULAD Experiment.ipynb* are used to conduct the mockup experiment. They use my custom modules in the same folders: *experiment_util.py*, *models.py*, and *plot_util.py*. *eval.config* is the Fairness Indicators configuration file to extract the desired result

- *experiment_util.py* provides global variables that can be used in the experiment and methods to transform the **DataFrame** object. It also offers the **Evaluation** class, which stores information about the datasets, the name of its column containing label value and sensitive attributes, the symbol representing the privileged groups, and separate sets of indexes for cross-validation.
    - Its methods *get_train_test* take a preprocessed dataset and the index of the set chosen for test fold and output training dataset and testing dataset retrieved from the preprocessed dataset
    - Its methods *framework_evaluate* take Series containing ground truth and models' prediction and output Series containing Accuracy metrics and bias metrics retrieved from the *extract_result* method.
    - Its methods *extract_result* take outputs from Fairness Indicators and extract its relevant information in the form of a Series
    - Its methods *cross_validation* take names of the models, training and testing methods from *modules.py*, and preprocessed datasets, use *get_train_test* to get training and testing sets for each epoch of cross-validation and feed them to the training and testing methods, then feed prediction and ground truth of each epoch to *framework_evaluate* to retrieve the epoch's metrics. In the end, it calculates the average of each metric and outputs them in a Series
- *models.py* provides training and testing methods for each accessed model algorithm
- *plot_util* provides a method to draw a scatter graph to visualise the results