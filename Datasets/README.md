# A preliminary analysis of tabular datasets

Here is a preliminary analysis of tabular datasets I will use to survey tabular bias evaluation frameworks. The goal of the analysis is to obtain a basic understanding of the characteristics of the datatsets (and their bias)

The metric I would use in this preliminary analysis will be:
- Number of instances in the dataset and number of features in the dataset
- The distribution of labels within the dataset, whether the dataset skews towards positive or negative
- The distribution of protected variables within the dataset, whether the dataset  skews towards protected or privileged
- The distribution of labels given certain protected variable values and which labels correlate more with certain protected variable groups.

The common protected variables I chose for the research are gender/sex.

## Datasets characteristic

The datasets I have collected can be categorised into two classes:

- Research datasets. These datasets are experimented with across multiple Machine Learning Fairness literature and are recognised as standard benchmark datasets in the community. They are generally well-maintained and represent a highly relevant ethical problem in machine learning applications.
    - [Adult](https://archive.ics.uci.edu/dataset/2/adult) has *48813* entries and *14* features. It has imbalanced labels and imbalanced gender representation. Females receive less representation, and their label distribution skews more towards negative than others.
    - [COMPAS](https://github.com/propublica/compas-analysis/tree/master) has *7214* entries and *51* features. It has relatively balanced labels and imbalanced gender representation. Females received less representation, and their label distribution skewed more towards negative than others.
    - [Diabetes](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) has *101766* entries *49* features. It has balanced labels and balanced gender representations. Lable distributions of both groups are similar to overall label distribution.
    - [German](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) has *1000* entries and *20* features. It has imbalanced labels and imbalanced gender representation. The female receives less representation, but label distributions of both groups are similar to overall label distribution.
    - [OULAD](https://analyse.kmi.open.ac.uk/open_dataset) has *29030* entries and *10* features. It has balanced labels and balanced gender representations. Lable distributions of both groups are similar to overall label distribution.

- Kaggle datasets. These datasets are sourced from Kaggle.com, a machine learning dataset hosting websites. They are binary classification problems with protected attributes (gender) involved. Compared to research datasets, their data quality is not guaranteed, and their problems may not be relevant. However, they present an application of bias mitigation methods unseen by literature.
    - [Job](https://www.kaggle.com/datasets/ayushtankha/70k-job-applicants-data-human-resource) has *73458* entries and *13* features. It has balanced labels but extremely imbalanced gender representation. Females receive fewer representations, but the label distributions of both groups are similar to the overall label distribution.
    - [College](https://www.kaggle.com/datasets/saddamazyazy/go-to-college-dataset) has *1000* entries and *10* features. It is a synthetic dataset. It has balanced labels and balanced gender representations. Lable distributions of both groups are similar to overall label distribution.
    - [Bank](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data) has *10000* entries and *10* featyres. It has imbalanced labels but balanced gender representation. Compared to the overall distribution, the label distribution of females skews more towards positive, while label distributions of males skew more towards negative
    - [Campus](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement) has *215* entries and *13* features. It has imbalanced labels and imbalanced gender representation. Lable distributions of both groups are similar to overall label distribution.
    - [Employee](https://www.kaggle.com/datasets/tawfikelmetwally/employee-dataset) has *2764* entries and *8* features. It has slightly imbalanced labels but balanced gender representation. The label distribution of females skews more towards positive than others.

## Summary of Datasets Characteristics

| **Datasets** | **Dataset Size** | **Feature sizes** | **Balanced label?** | **Balanced representation?** | **Additional Information**                                   |
|--------------|------------------|-------------------|---------------------|------------------------------|--------------------------------------------------------------|
| Adult        | 48813            | 14                | NO                  | NO                           | Female correlate with Negative                               |
| COMPAS       | 7214             | 51                | YES                 | NO                           | Female correlate with Negative                               |
| Diabetes     | 101766           | 49                | YES                 | YES                          | ----                                                         |
| German       | 1000             | 20                | NO                  | NO                           | ----                                                         |
| OULAD        | 29030            | 10                | YES                 | YES                          | ----                                                         |
| Job          | 73458            | 13                | YES                 | NO                           | ----                                                         |
| College      | 1000             | 10                | YES                 | YES                          | ----                                                         |
| Bank         | 10000            | 10                | NO                  | YES                          | Female correlate with Positive; Male correlate with Negative |
| Campus       | 215              | 13                | NO                  | NO                           | ----                                                         |
| Employee     | 2764             | 8                 | NO                  | YES                          | Female correlate with Positive                               |

## How to utilise this subrepo

This subrepo has been used in the Windows environment.

To use this repo in the Window Environment, use Python version 3.11 and [window_requirements.txt](./window_requirements.txt) with pip to install the virtual environment. 

It is also used for storing the datasets mentioned above. Research datasets are in *Research* folders, and Kaggle datasets are in *Kaggle* folders. **Adult**, **Diabetes** and **German** datasets are fetched using *fetch_ucirepo* methods from **ucimlrepo** packages.