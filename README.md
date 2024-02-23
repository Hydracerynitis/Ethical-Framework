# Bias Evaluation Framework Survey

## What is this Project?

Ethical Evaluation Framework Survey is my summer scholarship research project. The project aims to thoroughly examine available Bias Evaluation Frameworks, evaluate their strengths and weaknesses, and identify areas where future instalments may improve. To achieve this, I will try to learn to utilise each framework and set up a small benchmark experiment with them to demonstrate their capability. Afterwards, I will identify those frameworks' potential and limitations as model analysis tools in machine learning workflows based on my personal experience with them.

Nowadays, since machine learning models are increasingly applied to scenarios where decisions can alter an individual's life, the concern of whether those models make a fair prediction becomes stronger and stronger. As [study](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2018.3093) suggests, machine learning models can propagate bias in the training data and express them through their prediction. Fairness machine learning is an emerging research field that aims to tackle such an issue. Recently, they have developed bias metrics to quantify bias a model can express and bias mitigation methods to reduce such bias.

Bias Evaluation Frameworks are tools developed to allow users to calculate bias metrics of their models and understand the fairness of the model's prediction. In this project, I will examine frameworks suited for tabular data models and large language models. The framework analysis revolves around Reliability, Generalizability, Guidance and Robustness.

- Reliability means how far the framework can run without encountering any problems. Here, I record any errors or bugs I have encountered in my experience with the framework, including installing it, learning how to use it and running the experiment.
- Generalizability means how many other workflows the framework can be applied to. It means how much customisation the framework allows the users for calculation tasks. In other words, I record whether it allows users to use custom models, custom datasets, or even custom metrics.
- Guidance means how easy the framework is for new users to learn. Here, I record the availability of the framework's tutorial and any difficulty I have encountered when learning the framework
- Robustness means how good the framework is at evaluating the bias present in the model. Here, I examine the capability of the framework. In other words, I examine the framework's roster of available bias metrics and the delivery and visualisation of the framework's result to identify its strengths and limitations.

## Survey Result

## Framework for Tabular Models

### AIF360

The AI Fairness 360 Toolkit (AIF360) is a bias evaluation and mitigation framework for tabular data. It offers bias mitigation methods from the research knowledge sphere and bias evaluation functionality.

It is the baseline framework for comparison throughout this survey not only because 

#### Experiment

The experiment with AIF360 is conducted with Natania Thomas as a research project. The experiment is to examine the performance of its bias mitigation implementation in the case of Kaggle datasets rather than standard UCI datasets. The datasets we use are [Job](https://www.kaggle.com/datasets/ayushtankha/70k-job-applicants-data-human-resource), [Bank](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data) and [College](https://www.kaggle.com/datasets/saddamazyazy/go-to-college-dataset).

The bias mitigation implementation we accessed in the experiment includes two pre-process methods, two in-process methods and two post-process methods:
    - [Reweighting](http://doi.org/10.1007/s10115-011-0463-8) and [DisparateImpactRemover](https://doi.org/10.1145/2783258.2783311)
    - [AdverservialDebiasing](https://arxiv.org/abs/1801.07593) and [GridSearchReduction](https://arxiv.org/abs/1905.12843)
    - [CalibratedEqOddsPostprocessing](https://papers.nips.cc/paper/7151-on-fairness-and-calibration) and [RejectOptionClassification](https://doi.org/10.1109/ICDM.2012.45)

The main finding of our experiment is that the fair-accuracy tradeoff depends on the training dataset. The datasets in our experiment are too varied, so we can not find substantial evidence on the performance of bias mitigation methods. In addition, we find that fairness metrics had a limited impact on the comparison of fairness-accuracy tradeoffs between methods.

#### Potential and Limitation

AIF360 has the advantage of having a large amount of tutorial material for new users to utilise the tool and offering state-of-the-art bias mitigation implementation to improve their models. It also accepts custom models and custom datasets for bias evaluation and bias mitigation (other than in-processing methods).

However, AIF360's use of **StandardDataset** rather than **Dataframe** for its operation requires some rework to integrate it into an existing machine-learning workflow. In addition, **StandardDataset** lacks the flexibility that **Dataframe** offers, making some machine learning workflows incompatible with AIF360. AIF360 also lacks bias metrics based on predicted outcomes, similarity measures and causal reasoning. AIF360 may be less competitive in terms of bias mitigation due to its focus on in-processing methods, which, due to its nature, replace users' models rather than improve their models.

### Fairness Indicator

The fairness indicator is an additional library of tensorflow that provides a biased evaluation feature. It is built on the TensorFlow model analysis library while also packaging with the What-If tool, another standalone framework that allows the users to edit nodes in the model and present its consequence to the model's prediction.

#### Experiment

The experiment for fairness indicator is to investigate the impact of AIF360's [Disparate Impact Remover](https://doi.org/10.1145/2783258.2783311) implementation on scikit-learn's fairness unaware models, which include Random Forest, Multi-Layered Neural Network, and Gradient Boosting. The datasets I use are [Adult](https://archive.ics.uci.edu/dataset/2/adult), [OULAD](https://analyse.kmi.open.ac.uk/open_dataset), [Job](https://www.kaggle.com/datasets/ayushtankha/70k-job-applicants-data-human-resource), and [Bank](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data). I only used repair level 0.75 as the hyperparameter for Disparate Impact Remover throughout the experiment.

The main finding of our experiment is that Disparate Impact Remover works best in datasets with imbalanced label distribution, like OULAD and Bank. It works better when the dataset has a balanced protected attribute distribution. However, the fairness indicator only offers bias metrics based on the correctness of the model's prediction across protected and privileged groups, while Disparate Impact Remover is intended to reduce the difference in positive prediction between them. The findings of the experiment can be misleading.

#### Potential and Limitation

Since I could not learn and fully understand how to create and train a TensorFlow model during the project's timeframe, I can only evaluate the Fairness Indicator's strengths and weaknesses on non-tensorflow models. From what I can gather from the framework's information and its tutorial, I can only assume that it can provide better in-depth analysis and integration into TensorFlow workflow.

Fairness Indicator only needs the model's prediction, corresponding ground truth, and protected attribute from calculating the model's bias metrics. It also offers the user an interface to customise the framework's output through a config file (or writing config data in codes). It can be integrated easily into the model evaluation phase of the machine learning workflow and can be customised to cater to the user's requirements.

However, its bias metrics are very limited as it only focuses on visualising and comparing bias metrics between protected and privileged groups from a single prediction of a single model. To facilitate tasks comparing multiple predictions, such as cross-validation, practitioners have to write extra methods to extract data from the framework and process those data themselves. In addition, the bias metrics it offers are very narrow. As mentioned in the limitation of the experiment, it only offers metrics whose definitions are based on prediction and ground truth, making the framework meaningless in situations where such a definition of fairness does not apply.

### Aequitas

Aequitas is an open-source bias auditing and Fair Machine Learning toolkit. This package aims to provide an easy-to-use and transparent tool for auditing machine learning models, allowing users to correct models with bias mitigation methods and an automated benchmark system to find the models with the best performance-fairness tradeoff.

#### Experiment

The experiment investigates the effectiveness of Aequitas's **FairGBM** and **FairLearnClassifier** on performance-fairness-tradeoff. The comparison group in this experiment consist of scikit-learn's **RandomForestClassifier** and **GradientBoostingClassifier**. I only use one set of hyperparameters for **FairGBM** and **FairLearnClassifier**, which is based on the experiment config file for the framework's paper. The datasets used in the experiments are [Diabetes](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), [OULAD](https://analyse.kmi.open.ac.uk/open_dataset) and [College](https://www.kaggle.com/datasets/saddamazyazy/go-to-college-dataset).

The main findings of the experiment are that both **FairGBM** and **FairLearnClassifier** have varied performance-accuracy-tradeoff across all datasets, despite the datasets having a balanced label distribution and a balanced gender representation. In some cases, they provide great improvement in fairness that allows them to pass the significance test. In other cases, they overcorrect and worsen their bias metrics while trading away accuracy. It is assumed that there are other characteristics among those datasets that influence the effectiveness of **FairGBM** and **FairLearnClassifier**, which are worth investigating.

#### Potential and Limitation

Aequitas is one of the more advanced bias evaluation frameworks covered in this survey. Its audition feature provides an extensive range of bias metrics to meet the demands of users' bias evaluation framework. It also calculates and performs disparity-based fairness tests on those bias metrics. The result of the audition takes the form of **DataFrame**, making it easy to extract and transform information from. 

However, most bias metrics it provides have definitions based on prediction and ground truth, which makes aequitas less relevant in scenarios where fairness is more viewed as equity. In addition, aequitas's tutorial provides no rationale and explanation of the choice of each bias metric, making the roster of bias metrics overwhelming for users new to model fairness auditions. It is also worth noting that Aequitas focuses on evaluating a single prediction from a single model. Slightly more effort is required for users to conduct cross-validation and multiple model comparisons with aequitas. 

Aequitas also provides an automated benchmark system with in-depth available configuration options. This feature allows users to perform large-scale benchmarks on their custom fairness-aware models, with emphasis on their hyperparameters. However, it seems that it provides no interface to extract the exact hyperparameter of the best-performing model used.

## Framework for Language Models

### Evaluate

Evaluate is a language model evaluation framework by Hugging Face; it offers interfaces to calculate models' performance from their outputs. The bias metrics it offered include **Toxicity**, **Regard** and **Honest**

#### Experiment

The experiment is to test currently popular language models in Huggingface by evaluating and comparing them to each other on their ability to eliminate bias. The models I selected in the experiment are Mixtralai's Mixtral 7B, Microsoft's Phi-2, and TecentARC's LLaMA Pro 8B.

The experiment result shows that models from the least toxic to the most toxic are Phi-2, Mixtral 7B and LLaMa Pro 8B. However, the regard score also shows that Phi-2 has exhibited significant bias between females and males. It is more likely to make polarising comments regarding female individuals and make neutral comments regarding male individuals. It seems that Phi-2 has the least toxicity, which may be a result of overfitting.

Mixtral 7B and Phi-2 are incompatible with the Honest metrics. So, I have to expand the Honest experiment to get more information from LLaMa Pro 8B. Its result shows that LLaMa Pro 8B is capable of making less hurtful sentence completion regarding protected class (queer or female).

#### Potential and Limitation

The evaluation framework is a simple evaluation package that can be integrated into any language model workflow due to its lightweightedness. Since it requires text generation for calculation, utilising the framework can be a simple downstream task without altering the entire workflow too much. 

However, its ability to evaluate bias is very limited. **Toxicity**, **Regard**, and **Honest** all access the toxic aspects of the model without necessarily accessing the bias present in the model. In addition, **Regard** and **Honest** emphasize toxicity differences across different groups, making their findings on the model's bias much more restricted. It is beneficial for Evaluate to have many basic evaluation techniques on language models like WEAT, WEFAT or SEAT.

### Fairpy (LLM)

Fairpy (designed for language models, not to be confused with the framework with the same name targeting tabular models) is a framework that has gathered state-of-the-art bias evaluation methods on language models and bias mitigation methods. However, I am unable to recreate its functionality due to its lack of installation information and lack of backward compatibility with the newest technology. 

#### Potential and Limitation

Assuming those problems of the framework are fixed, it will be a serious contender for bias evaluation in language models. The bias metrics measurement, such as WEAT and SEAT, are not present in other available frameworks for language models, like Evaluate.
The Bias Evaluation and Bias Mitigation features are aggregated into two main classes, **BiasDetectionMetrics** and **BiasMitigationMethods**, making accessing them straightforward. However, it can enjoy some improvements. It can be made as a Python package with its dependency clearly stated, or at least have a readily available installation instruction. There also should be more guidance resources for new users to interpret and understand the bias metrics.

### HELM

(CRFM) HELM is a large-scale automatic benchmark framework whose operation is mainly focused on a console and uses a locally hosted webpage to deliver benchmark results. It supports 48 scenarios (datasets) to benchmark large language models (LLM) with core scenarios representing possible downstream applications of LLM and targeted evaluations designed to evaluate LLMs' skills and risk. The paper introducing the framework has benchmarked 36 state-of-the-art LLMs. It also offers some of the LLM models for users to set up their benchmark experiments. However, I am unable to replicate its result to a satisfactory degree to facilitate a benchmark experiment with it. 

#### Potential and Limitation

HELM has the capacity to perform an industry-wide audit of LLMs with its in-depth range of scenarios (datasets) and a large range of state-of-art large language models (LLM) it has aggregated. It is noted that it requires a dedicated Linux server with large storage space for large-scale benchmarking. In addition, currently, it only supports custom HuggingFace models while custom scenarios and custom local models are developed.

However, it may not be effective as a bias evaluation framework due to its main focus is not on evaluating bias. As a result, it has a limited number of metrics relevant to bias or fairness, which are perturbed accuracy regarding demographics (perturbed accuracy for short), demographic representation, stereotypical associations, and toxicity fraction. In addition, most scenarios do not calculate those metrics. Pertubated accuracy seems disconnected from other metrics, as it is usually not calculated with others at the same time. Furthermore, Demographic representation and stereotypical associations are count-based metrics, which are "brittle in several ways in general".

## Survey Conclusion

Bias evaluation frameworks covered in this survey can be categorised based on their strength and weaknesses as well as the developer's intent to develop them:
- Standard Framework. It consists of AI Fairness 360, Aequitas and Fairpy. These frameworks are developed by Fairness Machine Learning researchers aiming for the mass adaption of their findings. Consequently, these frameworks are equipped with state-of-the-art bias evaluation frameworks and bias mitigation implementations. Thus, they are capable of meeting the demands of users' fairness audition with their diverse range of bias metrics.
- Subordinate Framework. It consists of Fairness Indicators and Evaluate. These frameworks are minor libraries belonging to an existing major machine learning ecosystem (Tensorflow for fairness indicator and HuggingFace for Evaluate). They are likely developed to ensure the ecosystem has some form of response to the fairness problem. As a result, they have a small range of bias metrics, and their capacity for fairness benchmark experiments is very limited.
- Evaluation Framework. It is a variation of Standard Frameworks and only consists of HELM. This framework is developed by Fairness Machine Learning researchers to thoroughly evaluate language models, with Toxicity and Bias included as one of its topics. Consequently, it provides the extensive capability to handle users' demand for large-scale benchmarking. However, the bias metrics it offers are limited and weak compared to the Standard Framework.

The adaption of Subordinate Frameworks is relatively easy. However, its limited capability makes its mass adaption meaningless to address the fairness problem. Among the Standard Framework and Evaluation Framework, Aequitas is the most convenient to adapt due to it having less reliability and generalisation problems. However, HELM is the most suitable to adapt for users dealing with large-scale fairness evaluation due to it being large-scale oriented. 

One that is also worth pointing out is that all frameworks covered in this survey share different compatibility with technology. AI Fairness 360, Aequitas and Evaluate are capable of running on the newest version of Python (3.11) at the time of the survey. While Fairness Indicators, Fairpy and HELM are strictly limited to older versions of Python (3.8 and 3.9) and common dependencies.

The common problem that all bias evaluation frameworks have is that they provide limited and imbalanced bias metrics. Fairness Indicators, Evaluate, and HELM are only several metrics that can be categorised as bias metrics, which makes it hard to fully capture the extent of bias models are able to express. On the other hand, AI Fairness 360 and Aequitas have most of their bias metrics concerning models' prediction and respective ground truth. Consequently, they put more emphasis on fairness as equality than fairness as equity. Fairpy is an exception as its bias metrics can be considered to evaluate the model's casual reasoning (i.e. WEAT and SEAT) rather than simply predictions and ground truth. 

The other common problem I have commonly seen is that they give little guidance on how to use and interpret the metrics they offer. Their description of bias metrics and their definition are mostly mathematical. Their interpretation of certain bias metrics usually involves comparing them to bias metrics from other groups. As a result, all frameworks assume users have profound knowledge of the bias metrics and have a clear view of their fairness demands, allowing them to choose the metrics best suited for them. For users who are newly introduced to the fairness model audition, the framework's extensive range of bias metrics will be overwhelming for them to understand.

In future, the development of a bias evaluation framework should focus on creating and utilising bias metrics that are not based on models' predictions and the ground truth. [Study](https://dl.acm.org/doi/abs/10.1145/3411764.3445604) pointed out that we lack bias metrics that have their definition based on predicted probability and ground truth, similarity between predictions and causal reasoning. The frameworks should also improve on the guidance for choosing bias metrics. The guidance does not need to be very extensive. An example scenario or a line suggesting the metrics' application should be sufficient for new users.

Currently, there is no equivalent of AI Fairness 360 or Aequitas for bias evaluation frameworks on language models. More work can be done to develop a framework that not only offers diverse ranges of stronger bias metrics and state-of-the-art bias mitigation methods for language models but also has less reliability, generalisation and guidance problems.

## How to utilise this repository

Each subdirectory in this repository represents a subrepo that represents my research and experiment with bias evaluation frameworks except *Datasets* and *Additional Information*:
    - In each bias evaluation framework's subrepo, there is also a README.md, containing more detailed but less concise findings of the framework. There is also a section called **How to utilise this subrepo** containing information that allows you to replicate my experiment with it.
    - *Datasets* contain tabular datasets I gathered for my previous study and this survey. They (but not all of them) are used in the mockup experiment with bias evaluation frameworks on tabular models. There is a preliminary analysis of these datasets, containing information and characteristics of them.
    - *Additional Information* contains documents I created in preparation for this survey.
        - *Bias Detection algorithm_Tool.xlsx* record information of currently available artifacts I can find that are related to fairness audition. They contain links to the paper and GitHub repository of those artifacts as well as some basic description of that artefact. It also contains my review of a related Human Computer Interface study.
        - *Investigating the impact of bias mitigation methods.pdf* is the literature of my previous study. It can be an introductory read of this survey. I also upload my source code of this research to [GitHub](https://github.com/Hydracerynitis/Fairness-Metrics-Research)