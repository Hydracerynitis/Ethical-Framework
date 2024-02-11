# Fairness Indicator

Fairness Indicator is a bias evaluation framework that built upon Tensorflow Model Analysis. It position itself for tensorflow machine learning model workflow integration while also packaging with What-If widget.

## Experiment

I set up the an experiment with fairness indicator to investigate the effectiveness of [Disparate Impact Remover](https://doi.org/10.1145/2783258.2783311) (DIR) by AI Fairness 360. I will compare the difference between accuracy-bias-metrics-tradeoff of fairness unaware models and that of the same models with Disparate Impact Remover 

The fairness metrics in the experiments will be offered by the fairness indicator, which includes false positive rate, false negative rate, false discovery rate and false omission rate. To evalute the performance of Disparate Impact Remover, I will compare the change in those bias metrics difference between protected groups and privilleged groups.

The models I accessed in the experiment are Random Forest, Multi Layer Neural Network and Gradient Boosting models. For the metaparameters of DDisparate Impact RemoverIR, I only use 0.75 repair levels due to the timeframe the experiment take place.

The datasets I used in the experiment are [Adult](https://archive.ics.uci.edu/dataset/2/adult), [OULAD](https://analyse.kmi.open.ac.uk/open_dataset), [Job](https://www.kaggle.com/datasets/ayushtankha/70k-job-applicants-data-human-resource), and [Bank](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data) datasets. Such selction of datasets is to invesitgate the effectiveness of Disparate Impact Remover different cases of gender representation and gender distribution.

The characteristic of individual datasets are in the following: 
- Adult dataset has imbalanced label distribution and imbalanced gender representation. 
- OULAD dataset has balanced label distribution and balanced gender representation. 
- Job dataset has balanced label distribution and imbalanced gender representation.
- Bank dataset has imbalanced label distribution and balanced gender representation. 

## Experiment Result

After applying Disparate Impact Remover into the training process of Multi Layer Neutral Network  will result in the model only predicting negative. This occur across every dataset. Its false positive rate difference, false negative rate difference, false discovery rate difference and false omission rate difference become invalid and unfit for comparison. It is unclear what casue this issue. As a result, Neural Network model is excluded in the analysis 

Disparate Impact Remover makes slight difference in all models across all datasets. However, the effect of Disparate Impact Remover on bias metrics are varied depending on the training datasets. In case of Adult dataset, it does help improve models' False Positive Rate difference and False Negative Rate difference. However, it make Random Forest model's False Discovery Rate difference and False Omission Rate difference worse while making almost no change on Gradient Boosting model. In case of OULAD dataset and Job dataset, it either overcorrect models' bias metrics difference, or make them worse. And in case of Bank dataset, it either makes slight improvements or almost no change on bias metrics difference.

Overall, Disparate Impact Remover seem to work best when the training dataset has imbalanced label distribution. Having a balanced gender representation allows it to have a more effective improvement of model's fairness. Disparate Impact Remover is able to improve model's False Positive Rate Difference and False Negative Rate Difference. However, its effectiveness is limited when come to False Discovery Rate Difference and False Omission Rate Difference.

One thing has to be pointed out is that Disparate Impact Remover is intended to make model make similar amount of postive or negative predictions regardless the individual belongs to previllegd groups or protected groups. By its own definition, it is unfit to improve the model's overral correctness regarding previlleged groups and protected groups. As a result, the experiement's conclusion is limited and can be even considered misleading.

## Comment

### Reliability

- The Fairness Indicator is built upon Tensorflow Model Analysis, which is only available in Python 3.8 and 3.9. This installation preequisite is not mention on either the github page or tutorials.
- There are several instances of installing fairness indicators both on Window and Linux:
    - Installing the **fairness-indicators** of version > 0.40.0 on a newly created virtual environment (both on Python 3.8 and 3.9) will result in dependency conflict with its own depnedency **apache-beam**. This makes them uninstallable through pip.
        - This is due to **fairness-indicators**'s dependency **tensorflow-model-analysis** requires package **attrs** of version >=19.3.0 and <0.22, while **apache-beam**'s dependency **jsonschema** requires package **attrs** of version >=22.2.0
    - Installing **fairness-indicators** and **scikit-learn** (as well as **aif360** which depends on **scikit-learn**) will cause a dependency conflict
        - This is due to **fairness-indicators**'s dependency **tensorflow-data-validation** requires package **joblib** of version <0.15 and >=0.12 while **scikit-learn** (version 1.3.2) requires package **joblib** of version 1.3.2
        - This can be resolved by backtracking to older version of **scikit-learn** (<=1.0.2)
    - To correctly install the framework, you need to install its dependency first and then install the framework itself. Those information can be found on the framework's [github page](https://github.com/tensorflow/fairness-indicators)
- Throughout the experiment, I have encounter various problem that prevent me from using AI Fairness 360 implementations. However, it is unsure whether this is due to having older version python packages or due to bugs in the AIF360 implementations.
    - Affected implementations includes **AdversialDebiasing**, **GridSearchReduction**
- The framework will output string "NaN" when calculating the metrics when the model only make positive predictions or negative predictions. It will become error if the evalution pipeline does not take this into accounts.

### Generalizability

- A small thing to note is that the visualization of the framework assume the IDE theme user are using has white background. This has forced me to switch my IDE theme from black background to white background to be able to see the results.
- To produce a report of models' bias metric, it only needs its prediction and the correspounding ground truth and protected attribute values. The user can use a config file (or write configuration in code) to specify the columns of **DataFrame** to retrieve protected attribute, ground truth and models' prediction as well as the metrics it will calculate from them.


### Guidance

- It provides Colab notebooks tutorial to provide guidance for new users. However, the notebook is run on Python version 3.10. As Fairness-Indicators's preequisite is Python version 3.8 and 3.9, those notbooks tutorial are not currently working.
    - I have encountered a problem with the framework that can be attribute to dependecy version conflict, where the framework exihibit different behaviour that resulting in an error.

### Robustness

- One thing to note is that I only evaluate the framework's performance regarding non-tensorflow models.Since I was unable to learn and fully understand how to create and train a tensorflow model during the timeframe of the project. 
    - Since the framework is built on tensorflow libary, I assume that it could provide a more in-depth and better analysis of tensorflow model than non-tensorflow model as well as better integration into tensorflow model workflow.
- Fairness Indicator framework can be integrated easily into the model evaluation phase of the machine learning workflow and can be customized catering to user's requirements.
- However The framework is built for comparison between performance of privilleged groups and protected groups from a single prediction of single model. It also lack the capability to calculate metrics of model's overall performance
    - It's very difficult to extract data from the framework for further analysis, extra methods reuqired for user for processing cross-validation results and multiple model comparison
- The definitions of available metrics consist of only those based on predicted outcomes and ground truth. The framework also present its data in form of comparisons between privilleged groups and protected groups.
    - This make framework much weaker in situations where ensuring the model achieve equal accuracy or other metrics is not ideal or even not possible. (i.e. Disaparate Impact)