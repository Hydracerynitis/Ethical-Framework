# Ethical Framework

This is the repository for my summer scholarship research, which test currently available Bias Evaluation Frameworks by setting up an experiments with the frameworks and examine its adavantage and limitation in the role of a machine learning partitioners. The expected outcome of my research will be a report detialing the capability of current rolls of Bias Evaluation Frameworks and the knowledge gap where those tools can be improved upon.

# AIF360

AI Fairness 360 Toolkit (AIF360) is a bias evaluation and bias mitigation framework for tabular data. It offers bias mitigation methods from research knowledge sphere as well as bias evaluation functionality.

## Experiment

The experiment with AIF360 is conducted with Natania Thomas as a research project. The experiment is to examine the performance of its bias mitigation implementation in the case of Kaggle datasets rather than standard UCI datasets. The datasets we use is [Job](https://www.kaggle.com/datasets/ayushtankha/70k-job-applicants-data-human-resource), [Bank](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data) and [College](https://www.kaggle.com/datasets/saddamazyazy/go-to-college-dataset).

The bias mitigation implementation we accessed in the experiment includes 2 pre-process methods, 2 in-process methods and 2 post-process methods:
    - [Reweighting](http://doi.org/10.1007/s10115-011-0463-8) and [DisparateImpactRemover](https://doi.org/10.1145/2783258.2783311)
    - [AdverservialDebiasing](https://arxiv.org/abs/1801.07593) and [GridSearchReduction](https://arxiv.org/abs/1905.12843)
    - [CalibratedEqOddsPostprocessing](https://papers.nips.cc/paper/7151-on-fairness-and-calibration) and [RejectOptionClassification](https://doi.org/10.1109/ICDM.2012.45)

The main finding of our experiment is that the fair-accuracy tradeoff is dependent on the training dataset. And the datasets in our experiment are too varied, we can not find a substantial evidence on performance of bias mitigation methods. In addition, we find that fairness metrics had limited impact on comparison of fairness-accuracy tradeoff between methods.

## Potential and Limitation

AIF360 has the great of advantage of having the large amount of tutorial material for new partitioners to utilise tool, as well as offering state-of-art bias mitigation implementation for them to improve their models. It also accepts custom models and custom datasets for bias evaluation and bias mitigation (other than in-processing methods).

However AIF360's use of **StandardDataset** rather than **Dataframe** for its operation meanse that it require some rework to be done to integerate it into an exisingt machine learning workflow. In addition, **StandardDataset** lack the flexibility that **Dataframe** offers, making some machine learning workflow incompatible with AIF360. AIF360 also suffer from lack of bias metrics that based on predicited outcome, similarity measures and casual reasoning. AIF360 may be less compatitive in terms of bias mitigation due to it foucs on in-processing methods which due to its nature replace partitioners' model rather than improve their model.

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