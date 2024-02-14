# Bias Evaluation Framework Survey

## What is this Project?

Ethical Evaluation Framework Survey is my summer scholarship research project. The goal of the project is to thoroughly examin currently available Bias Evalutation Frameworks, evaluate their strength and weakness, and identify areas where future installments may improve upon. In order to achieve it, I will try to learn utilizing each individual frameworks and set up a small benchmark experiment with them to demonstrate their capability. Afterward, I will identify those framework's potential and limitation as a model analysis tools in machine learning workflows based on my personal experience with them.

Nowadys since machine learning mdoels are increasingly applyed to scenarios where decsions are able to alter individual's life, the concern of whether those models are making a fair prediction becomes stronger and stronger. As [study](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2018.3093) suggest, machine learning models can propogate bias present in the training data and then express them through their prediction. Fairness machine learning is an emerging research field which aims to tackle such issue. Recently, they have developed bias metrics to quantify bias a model can express and bias mitigation methods to reduce such bias.

Bias Evaluation Frameworks are tools developed to offer paritioners the ability to calculate bias metrics of their models and understand the fairness of the model's prediction. In this project, I will examine two sets of frameworks, ones suited for tabular data models and ones suited for large language model. The analysis of the framework revolves aorund Reliability, Generalizability, Guidance and Robustness.

- Reliability means how far the framework can run without encoutering any problem. Here I record any errors or bugs I have encountered in my experience with the framework, including installing the framework, learning how to use it and running the experiment.
- Generalizability means how many other workflows can the frame be applied to. It means that how much customization the framework allows the users to have for the calculation tasks. In other words, here I record whether it allows user to use custom models, custom datasets, or even custom metrics.
- Guidance meanse how easy the framework is for new users to learn. Here I record the availablity of the framework's tutorial and any difficulty I have encountered when learning the framework
- Robustness meanse how good the framework is at evaluating the bias present in the model. Here I examine the capability of the framework. In other words, I examine the framework's roster of available bias metrics as well as the delievery and visualisation of framework's result to identify its strength and limitation.

## Survey Result

## Framework for Tabular Models

### AIF360

AI Fairness 360 Toolkit (AIF360) is a bias evaluation and bias mitigation framework for tabular data. It offers bias mitigation methods from research knowledge sphere as well as bias evaluation functionality.

It is the baseline framework for comparison throughout this survey not only because 

#### Experiment

The experiment with AIF360 is conducted with Natania Thomas as a research project. The experiment is to examine the performance of its bias mitigation implementation in the case of Kaggle datasets rather than standard UCI datasets. The datasets we use are [Job](https://www.kaggle.com/datasets/ayushtankha/70k-job-applicants-data-human-resource), [Bank](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data) and [College](https://www.kaggle.com/datasets/saddamazyazy/go-to-college-dataset).

The bias mitigation implementation we accessed in the experiment includes 2 pre-process methods, 2 in-process methods and 2 post-process methods:
    - [Reweighting](http://doi.org/10.1007/s10115-011-0463-8) and [DisparateImpactRemover](https://doi.org/10.1145/2783258.2783311)
    - [AdverservialDebiasing](https://arxiv.org/abs/1801.07593) and [GridSearchReduction](https://arxiv.org/abs/1905.12843)
    - [CalibratedEqOddsPostprocessing](https://papers.nips.cc/paper/7151-on-fairness-and-calibration) and [RejectOptionClassification](https://doi.org/10.1109/ICDM.2012.45)

The main finding of our experiment is that the fair-accuracy tradeoff is dependent on the training dataset. And the datasets in our experiment are too varied, we can not find a substantial evidence on performance of bias mitigation methods. In addition, we find that fairness metrics had limited impact on comparison of fairness-accuracy tradeoff between methods.

#### Potential and Limitation

AIF360 has the great of advantage of having the large amount of tutorial material for new users to utilise tool, as well as offering state-of-art bias mitigation implementation for them to improve their models. It also accepts custom models and custom datasets for bias evaluation and bias mitigation (other than in-processing methods).

However AIF360's use of **StandardDataset** rather than **Dataframe** for its operation meanse that it require some rework to be done to integerate it into an exisingt machine learning workflow. In addition, **StandardDataset** lack the flexibility that **Dataframe** offers, making some machine learning workflow incompatible with AIF360. AIF360 also suffer from lack of bias metrics that based on predicited outcome, similarity measures and casual reasoning. AIF360 may be less compatitive in terms of bias mitigation due to it foucs on in-processing methods which due to its nature replace users' model rather than improve their model.

### Fairness Indicator

Fairness Indicator is an additional library of tensorflow which provide bias evalution feature. It is built on tensorflow model analysis library while also packaging with What-If tool, another standalone framework that allowing user to edit nodes in the model and present its consequence to the model's prediction.

#### Experiment

The experiment for fairness indicator is to investigate the impact of AIF360's [Disparate Impact Remover](https://doi.org/10.1145/2783258.2783311) implementation on scikit-learn's fairness unaware models, which include Random Forest, Multi-Layered Neural Network, and Gradient Boosting. The datasets I use are [Adult](https://archive.ics.uci.edu/dataset/2/adult), [OULAD](https://analyse.kmi.open.ac.uk/open_dataset), [Job](https://www.kaggle.com/datasets/ayushtankha/70k-job-applicants-data-human-resource), and [Bank](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data). I only use repair level 0.75 as the hyperparameters for Disparate Impact Remover throughout the experiment.

The main finding of our experiment is that Disparate Impact Remover work best in datasets that have imbalanced label distribution like OULAD and Bank. It work better when the dataset has balanced protected attribute distribution. However, the fairness indicator only offer bias metrics based on the correctness of the model's prediction across protected and privilleged groups while Disparate Impact Remover is inended to reducreduceing the difference of positive prediction between them. The findings of the experiment can be misleading.

#### Potential and Limitation

Since I was unable to learn and fully understand how to create and train a tensorflow model during the timeframe of the project, I can only evaluate Fairness Indicator's strength and weakness on non-tensorflow models. From what I can gather on the framework's infomation and its tutorial, I can only assume that it can provide better in-depth analysis as well as better integration into tensorflow workflow.

Fairness Indicator only needs model's prediction and the correspounding ground truth and protected attribute to calculate model's bias metrics. It also offer user an interface to customize the framework's output through config file (or writing config data in codes). It means that it can be integrated easily into the model evaluation phase of the machine learning workflow and can be customized catering to user's requirements

However, its bias metrics are very limited as it only focus on visualing and comparing bias metrics between protected groups and privilleged groups from a single prediction of a single model. To facilitate tasks comparing mulitple predictions such as cross validation, paritioners have to write extra methods to extract data from the framework and process those data themselves. In addition, the bias metrics it offers are very narrow. As mentioned in the limitation of the experiment, it only offer metrics which definition based on predicition and ground truth, making the framework meaningless in situation where such deifinition of fairness does not apply.

### Aequitas

Aequitas is an open-source bias auditing and Fair Machine Learning toolkit. This package aims to provide an easy-to-use and transparent tool for auditing machine learning models, allowing users to correct models with bias mitigation methods and an automated benchmark system to find the models with the best performance-fairness tradeoff.

#### Experiment

The experiment is to investigate the effectiveness of Aequitas's **FairGBM** and **FairLearnClassifier** on performance-fairness-tradeoff. The comparison group in this experiment consist of scikit-learn's **RandomForestClassifier** and **GradientBoostingClassifier**. I only use one set of hyperparameters for **FairGBM** and **FairLearnClassifier**, which is based on the experiment config file for the framework's paper. The datasets used in the expeirments are [Diabetes](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), [OULAD](https://analyse.kmi.open.ac.uk/open_dataset) and [College](https://www.kaggle.com/datasets/saddamazyazy/go-to-college-dataset).

The main findings of the experiment is that both **FairGBM** and **FairLearnClassifier** has varied performance-accuracy-tradeoff across all datasets, despite the datasets having a balanced label distribution and a balanced gender representation. In some cases, they provide great improve on fairness that allowing them to pass the significance test. In other cases, they overcorrect and worsen their bias metrics while trading away accuracy. It is assumed that there are other characteristics among those datasets influence the effectiveness of **FairGBM** and **FairLearnClassifier**, which are worth investigating.

#### Potential and Limitation

Aequitas is one of the more advanced bias evaluation framework covered in this survey. Its audition feature provide an extensive range of bias metrics to meet demands of users' bias evaluation framework. It also calculate and perform disparity based fairness test on those bias metrics. The result of the audition take form of **DataFrame**, making it easy to extract and transform information from. 

However, most bias metrics it provides have definition based on prediction and ground truth, makes aequitas less relevant in scenarios where fairness is more viewed as equity. In addition, aequitas's tutorial provide no rationale and explaination of the choice of each bias metrics, making the roster of bias metrics overwhelming for users new to model fairness auditions. It is also worth noting that aequitas focuses on evaluation single prediciton from a single model. Slightly more efforts are required for users to conduct cross-validation and mulitple models comparison with aequitas. 

Aequitas also provides automated benchmark system with in-depth avaible configuration options. This feature allows users to perform large scale benchmark on their custom fairness-aware models, with emphasis on their hyperparameters. However, it seems that it provide no interface to extract the exact hyperparameter the best performing model used.

## Framework for Language Models

### Evaluate

Evaluate is a language model evaluation framework by Hugging Face, it offer interfaces to calculate models' performance from their outputs. The bias metrics it offerred include **Toxicity**, **Regard** and **Honest**

#### Experiment

The experiment is to test currently popular language models in Huggingface with Evaluate and compare them each other on their ability to eliminate bias. The Model I selected in the experimented are Mixtralai's Mixtral 7B, Microsoft's Phi-2, TecentARC's LLaMA Pro 8B

The experiment result shows that models from the least toxic to the most toxic are Phi-2, Mixtral 7B and LLaMa Pro 8B. However, the regard score also show that Phi-2 has exihibited signigicant bias between female and males. It more likely to make polarising comments regarding female individuals and make neutral comments regarding male individuals. It seems that Phi-2 has least toxicity may be a result of overfitting.

Mixtral 7B and Phi-2 is incompatible with the Honest metrics. So I have to expand the Honest experiment to get more information from LLaMa Pro 8B. Its result shows that LLaMa Pro 8B is capable to make less hurtful sentence completion regarding protected class (queer or female).

#### Potential and Limitation

Evaluate framework is a simple evaluation package that can be integrated any language model workflow due to its lightweightness. Since require text generation for calculation, utilizing the framework can be a simple downstream task without altering the entire workflow too much. 

However, its ability to evaluate bias is very limited. **Toxicity**, **Regard** and **Honest** all access the toxic aspects of the model without necessarily accessing the bias present in the model. In addition, **Regard** and **Honest** put emphasis on toxicity difference across different groups, making its finding on model's bias much more restricted. It is benificial for Evaluate to have manay bais evaluation techniques on language models like WEAT, WEFAT or SEAT.

### Fairpy (LLM)

Fairpy (designed for language model, not to confuse with the framework with same name targeting tabular models) is a framework that has gathered state-of-art language model bias evaluation methods and bias mitigation methods. However, I am unable to recreate its functionality due to its lack of installation information and lack of backword compatibility with newest technology. 

#### Potential and Limitation

Assuming those problems of the framework are fixed, it will be a serious contender for bias evaluation in language models. The bias metrics measurement, such as WEAT and SEAT, are not present in other available frameworks for lanaguage models, like Evaluate.
The Bias Evaluation features and Bias Mitigation features are aggregated into two main class **BiasDetectionMetrics** and **BiasMitigationMethods**, making accessing to them simple and straightforward. However, it can enjoy some improvements. It can be made as a Python packages with its dependency clearly stated, or at least have a readily available installation instruction. There also should be more guidance resource for new users to interprete and understand the bias metrics.

### HELM

(CRFM) HELM is a large scale automatic benchmark framework with its operation mainly focused on console and use a localy hosted webpage to deliver benchmark result. It supports 48 scenarios (datasets) to benchmark large language model (LLM) with core scenarios representing possible downstream applications of LLM, and yargeted evaluations designed to evaluate LLMs' skills and risk. In the paper introducing the framework, it has benchmarked 36 state-of-art LLM. It also offers some of the LLM models for users to set up their own benchmark experiment. However, I am unable to replicate its result to a satisfactory degree to facilitate a benchmark experiment with it. 

#### Potential and Limitation

HELM has the capacity to perform an indsutry-wide audit of LLMs with its in-depth range of scenarios (datasets) and large range of state-of-art large language models (LLM) it has aggregated. It is noted that it require a dedicate Linux server with large storage space for large scale benchmarking. In addtion, currently it only support custom HuggingFace models while custom scenarios and custom local models are developing.

However, it may not be effective as a bias evaluation framework due to its main focus is not on evaluting bias. As a result, it has limited numbers of metrics relevant to bias or fairness, which are pertubated accuracy regarding demographic (pertubated accuracy for short), demographic representation, stereotypical associations, and toxicity fraction. In addition, most scenarios do not calucate those metrics. Pertubated accuracy seems disconnected with other metrics, as it is usally not calculated with others at the same time. Furthermore, Demographic representation and stereotypical associations are count-based metrics, which is "brittle in several ways in general".

## Survey Conclusion

Bias evalution frameworks covered in this survey can be catorigised based on their strength and weakness as well as the developer's intent to develop them:
- Standard Framework. It consists of AI Fairness 360, Aequitas and Fairpy. These frameworks are developed by Fairness Machine Learning researchers aiming for the mass adaption of their findings. Consequently, these frameworks are equiped with state-of-the-art bias evaluation framework and bias mitigation implementations. Thus, they are capable to meet the demands of users' fairness audition with their diverse range of bias metrics.
- Subordinate Framework. It consists of Fairness Indicators and Evaluate. These frameworks are minor libaries belonging to an existing major machine learning ecosystem (Tensorflow for fairness indicator, and HuggingFace for Evaluate). They are likely developed to ensure the ecosystem have some form of response to the fairness problem. As a reuslt, they have small range of bias metrics and their capacity for fairness benchmark experiments is very limited.
- Evaluation Framework. It is a variation of Standard Framework and only consists of HELM. This framework is develoepd by Fairness Machine Learning researchers to thoroughly evaluate language models, with Toxicity and Bias inclued as one of its topic. Consequently, it provides extensive capability to handle users' demand for a large scale benchmarking. But the bias metrics it offers are limited and weak compared to Standar Framework.

The adaption of Subordiante Frameworks are relatively easy. However, its limited capability makes its mass adaption meaningless to address the fairness problem. Among Standard Framework and Evaluation Framework, Aequitas is the most conveniet to adapt due to it has less reliability and generalisation problem. However, HELM is the most suitable to adapt for users dealing with large scale fairness evaluation due to it being large-scale-benckmark oriented. 

One that is also worth pointing out is that all frameworks covered in this survey share different compatibility with technology. AI Fairness 360, Aequitas and Evaluate is capable to run on the newest version of Python (3.11) at the time of the survey. While Fairness Indicators, Fairpy and HELM are strictly limited to older version of Python (3.8 and 3.9) and common dependencies.

The common problem that all bias evaluation frameworks have is that they provides limited and imbalanced bias metrics. Fairness Indicators, Evaluate and HELM only sevaral metrics that can be categroised as bias metrics, which makes them hard to fully capture the extent of bias models are able to express. On the other hand, AI Fairness 360 and Aequitas have most of their bias metrics concerning models' prediction and respective ground truth. Consequently, they put more emphasis on fairness as equality than fairness as equity. Fairpy is an exception as its bias metrics can be considered to evalaute model's casual reasoning (i.e WEAT and SEAT) rather than simply predictions and ground truth. 

The other common problem I have commonly seen is that they give little guidance on how to use and interprete the metrics they offer. Their description of bias metrics and their definition are mostly mathematical. Their interpretation of certain bias metrics usually involves comparing them to bias metrics from other group. As a result, all frameworks assume users have profound knowledge with the bias metrics and have clear view of their fairness demands, allowing them to choose the metrics best suited for them. For users that are newly introduced to fairness model audition, the framework's extensive range of bias metrics will be overwhelming for them to understand.

In future, the development of bias evaluation framework should focus on creating and utilizing bias metrics that are not based on models' predictions and the ground truth. [Study](https://dl.acm.org/doi/abs/10.1145/3411764.3445604) had pointed out that we lack bias metrics that have their definition based on predicted probability and ground truth, similarity between predictiosn and casual reasoning. The frameworks should also improve on the guidance for choosing bias metrics. The guidance does not need to be very extensive，an example scenario or a line suggesting the metrics' application should be sufficient for new users.

Currently there is no equivalend of AI Fairness 360 or Aequitas for bias evaluation frameworks on language models. More work can be don to develop a framework that not only offer diverse ranges of stronger bias metrics and state-of-the-art bias mitigation methods for langauge models, but also have less reliability, generalisation and guidance problems.

## How to utilise this repository

Each subdirectory in this repository represent a subrepo that reprenset my research and experiment with bias evaluation frameworks except *Datasets* and *Additional Information*:
    - In each bias evaluations frameworks' subrepo, there is also a README.md, containing a more detailed but less concise findings of the framework. There is also a section called **How to utilise this subrepo** containing informations that allows you to replicate my experiment with it.
    - *Datasets* contains tabular datasets I gather for my previous study and for this survey. They (but not all of them) are used in the mockup experiment with bias evaluation frameworks on tabular models. There is a preliminary analysis of these datasets, containing information and characteristics of them.
    - *Additional Information* contains documents I created in prepare for this survey.
        - *Bias Detection algorithm_Tool.xlsx* record informations of currently available artifacts I can find that are related to fairness audtion. They contain links to paper and github repository of those artifacts as well as some basic description of those artefact. It also contains my review of a related Human Computer Interface study.
        - *Investigating the impact of bias mitigation methods.pdf* is the literature of my previous study. It can be an introductory read of this survey. I also upload my source code of this research to [GitHub](https://github.com/Hydracerynitis/Fairness-Metrics-Research)