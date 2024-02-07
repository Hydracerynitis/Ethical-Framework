# Ethical Framework Evaluation Research Project

This is the repository for my summer scholarship research, which test currently available Bias Evaluation Frameworks by setting up an experiments with the frameworks and examine its adavantage and limitation in the role of a machine learning partitioners. The expected outcome of my research will be a report detialing the capability of current rolls of Bias Evaluation Frameworks and the knowledge gap where those tools can be improved upon.

# AIF360

AI Fairness 360 Toolkit (AIF360) is a bias evaluation and bias mitigation framework for tabular data. It offers bias mitigation methods from research knowledge sphere as well as bias evaluation functionality.

It is the baseline framework for comparison during this research.

## Experiment

The experiment with AIF360 is conducted with Natania Thomas as a research project. The experiment is to examine the performance of its bias mitigation implementation in the case of Kaggle datasets rather than standard UCI datasets. The datasets we use are [Job](https://www.kaggle.com/datasets/ayushtankha/70k-job-applicants-data-human-resource), [Bank](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data) and [College](https://www.kaggle.com/datasets/saddamazyazy/go-to-college-dataset).

The bias mitigation implementation we accessed in the experiment includes 2 pre-process methods, 2 in-process methods and 2 post-process methods:
    - [Reweighting](http://doi.org/10.1007/s10115-011-0463-8) and [DisparateImpactRemover](https://doi.org/10.1145/2783258.2783311)
    - [AdverservialDebiasing](https://arxiv.org/abs/1801.07593) and [GridSearchReduction](https://arxiv.org/abs/1905.12843)
    - [CalibratedEqOddsPostprocessing](https://papers.nips.cc/paper/7151-on-fairness-and-calibration) and [RejectOptionClassification](https://doi.org/10.1109/ICDM.2012.45)

The main finding of our experiment is that the fair-accuracy tradeoff is dependent on the training dataset. And the datasets in our experiment are too varied, we can not find a substantial evidence on performance of bias mitigation methods. In addition, we find that fairness metrics had limited impact on comparison of fairness-accuracy tradeoff between methods.

## Potential and Limitation

AIF360 has the great of advantage of having the large amount of tutorial material for new partitioners to utilise tool, as well as offering state-of-art bias mitigation implementation for them to improve their models. It also accepts custom models and custom datasets for bias evaluation and bias mitigation (other than in-processing methods).

However AIF360's use of **StandardDataset** rather than **Dataframe** for its operation meanse that it require some rework to be done to integerate it into an exisingt machine learning workflow. In addition, **StandardDataset** lack the flexibility that **Dataframe** offers, making some machine learning workflow incompatible with AIF360. AIF360 also suffer from lack of bias metrics that based on predicited outcome, similarity measures and casual reasoning. AIF360 may be less compatitive in terms of bias mitigation due to it foucs on in-processing methods which due to its nature replace partitioners' model rather than improve their model.

# Fairness Indicator

Fairness Indicator is an additional library of tensorflow which provide bias evalution feature. It is built on tensorflow model analysis library while also packaging with What-If tool, another standalone framework that allowing partitioner to edit nodes in the model and present its consequence to the model's prediction.

## Experiment

The experiment for fairness indicator is to investigate the impact of AIF360's [Disparate Impact Remover](https://doi.org/10.1145/2783258.2783311) implementation on scikit-learn's fairness unaware models, which include Random Forest, Multi-Layered Neural Network, and Gradient Boosting. The datasets we use are [Adult](https://archive.ics.uci.edu/dataset/2/adult), [OULAD](https://analyse.kmi.open.ac.uk/open_dataset), [Job](https://www.kaggle.com/datasets/ayushtankha/70k-job-applicants-data-human-resource), and [Bank](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data). I only use repair level 0.75 as the metaparameters for Disparate Impact Remover throughout the experiment.

The main finding of our experiment is that Disparate Impact Remover work best in datasets that have imbalanced label distribution like OULAD and Bank. It work better when the dataset has balanced protected attribute distribution. However, the fairness indicator only offer bias metrics based on the correctness of the model's prediction across protected and privilleged groups while Disparate Impact Remover is inended to reducreduceing the difference of positive prediction between them. The findings of the experiment can be misleading.

## Potential and Limitation

Since I was unable to learn and fully understand how to create and train a tensorflow model during the timeframe of the project, I can only evaluate Fairness Indicator's strength and weakness on non-tensorflow models. From what I can gather on the framework's infomation and its tutorial, I can only assume that it can provide better in-depth analysis as well as better integration into tensorflow workflow.

Fairness Indicator only needs model's prediction and the correspounding ground truth and protected attribute to calculate model's bias metrics. It also offer partitioner an interface to customize the framework's output through config file (or writing config data in codes). It means that it can be integrated easily into the model evaluation phase of the machine learning workflow and can be customized catering to partitioner's requirements

However, its bias metrics are very limited as it only focus on visualing and comparing bias metrics between protected groups and privilleged groups from a single prediction of a single model. To facilitate tasks comparing mulitple predictions such as cross validation, paritioners have to write extra methods to extract data from the framework and process those data themselves. In addition, the bias metrics it offers are very narrow. As mentioned in the limitation of the experiment, it only offer metrics which definition based on predicition and ground truth, making the framework meaningless in situation where such deifinition of fairness does not apply.

# Evaluate

Evaluate is a language model evaluation framework by Hugging Face, it offer interfaces to calculate models' performance from their outputs. The bias metrics it offerred include **Toxicity**, **Regard** and **Honest**

## Experiment

The experiment is to test currently popular language models in Huggingface with Evaluate and compare them each other on their ability to eliminate bias. The Model I selected in the experimented are Mixtralai's Mixtral 7B, Microsoft's Phi-2, TecentARC's LLaMA Pro 8B

The experiment result shows that models from the least toxic to the most toxic are Phi-2, Mixtral 7B and LLaMa Pro 8B. However, the regard score also show that Phi-2 has exihibited signigicant bias between female and males. It more likely to make polarising comments regarding female individuals and make neutral comments regarding male individuals. It seems that Phi-2 has least toxicity may be a result of overfitting.

Mixtral 7B and Phi-2 is incompatible with the Honest metrics. So I have to expand the Honest experiment to get more information from LLaMa Pro 8B. Its result shows that LLaMa Pro 8B is capable to make less hurtful sentence completion regarding protected class (queer or female).

## Potential and Limitation

Evaluate framework is a simple evaluation package that can be integrated any language model workflow due to its lightweightness. Since require text generation for calculation, utilizing the framework can be a simple downstream task without altering the entire workflow too much. 

However, its ability to evaluate bias is very limited. **Toxicity**, **Regard** and **Honest** all access the toxic aspects of the model without necessarily accessing the bias present in the model. In addition, **Regard** and **Honest** put emphasis on toxicity difference across different groups, making its finding on model's bias much more restricted. It is benificial for Evaluate to have manay bais evaluation techniques on language models like WEAT, WEFAT or SEAT.

# Fairpy (LLM)

Fairpy (designed for language model, not to confuse with the framework with same name targeting tabular models) is a framework that has gathered state-of-art language model bias evaluation methods and bias mitigation methods. However, I am unable to recreate its functionality due to its lack of installation information and lack of backword compatibility with newest technology. 

## Potential and Limitation

Assuming those problems of the framework are fixed, it will be a serious contender for bias evaluation in language models. The bias metrics measurement, such as WEAT and SEAT, are not present in other available frameworks for lanaguage models, like Evaluate.
The Bias Evaluation features and Bias Mitigation features are aggregated into two main class **BiasDetectionMetrics** and **BiasMitigationMethods**, making accessing to them simple and straightforward. However, it can enjoy some improvements. It can be made as a Python packages with its dependency clearly stated, or at least have a readily available installation instruction. There also should be more guidance resource for new partitioners to interprete and understand the bias metrics.

# HELM

(CRFM) HELM is a large scale automatic benchmark framework with its operation mainly focused on console and use a localy hosted webpage to deliver benchmark result. It supports 48 scenarios (datasets) to benchmark large language model (LLM) with core scenarios representing possible downstream applications of LLM, and yargeted evaluations designed to evaluate LLMs' skills and risk. In the paper introducing the framework, it has benchmarked 36 state-of-art LLM. It also offers some of the LLM models for partitioners to set up their own benchmark experiment. However, I am unable to replicate its result to a satisfactory degree to facilitate a benchmark experiment with it. 

## Potential and Limitation

HELM has the capacity to perform an indsutry-wide audit of LLMs with its in-depth range of scenarios (datasets) and large range of state-of-art large language models (LLM) it has aggregated. It is noted that it require a dedicate Linux server with large storage space for large scale benchmarking. In addtion, currently it only support custom HuggingFace models while custom scenarios and custom local models are developing.

However, it may not be effective as a bias evaluation framework due to its main focus is not on evaluting bias. As a result, it has limited numbers of metrics relevant to bias or fairness, which are pertubated accuracy regarding demographic (pertubated accuracy for short), demographic representation, stereotypical associations, and toxicity fraction. In addition, most scenarios do not calucate those metrics. Pertubated accuracy seems disconnected with other metrics, as it is usally not calculated with others at the same time. Furthermore, Demographic representation and stereotypical associations are count-based metrics, which is "brittle in several ways in general".