# Evaluate

Evaluate is a language model evaluation framework by Hugging Face, it offer interfaces to calculate models' performance from their outputs. The bias metrics it offerred include **Toxicity**, **Regard** and **Honest**

## Experiment

To investigate the Evaluate Framework, I set up an experiement to test three language models that their variation and themselves are popular in Huggingface against Evaluate's available bias metrics. The models I choose are Mixtralai's 7 Billion parameters [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) model, Microsoft's 2.4 Billion parameters [Phi-2](https://huggingface.co/microsoft/phi-2) model and Meta's [LLaMa](https://huggingface.co/meta-llama/Llama-2-7b) model. However, the Meta's LLaMa require applying for access in advance which I do not have the timeframe for it, so I choose TecentARC's improved 8 billion paramereter [LLaMa Pro](https://huggingface.co/TencentARC/LLaMA-Pro-8B) model for substitute.

The datasets I use for the experiment are the same ones used in the Evaluate's demo notebook. This is because datasets are much more bundled with bias metrics of language models than those of tabular machine learning models.

## Experiment Results

### Toxicity

Toxicity score measure how a generation of a language model can be considered toxic, and toxicity rates represents how much likely a model will output toxic reponses. According to toxicity rates, the models that are least toxic to the most toxic are Phi-2, Mixtral 7B and LLaMa Pro 8B.

Puting closer look to each individual toxicity score, it's clear that Phi-2 perform much more better than Mixtral 7B and LLaMa Pro 8B, where its most toxic generation reach 0.4 toxicity score while those of other models reach 1 toxicity core. Mixtral 7B and LLaMa Pro 8B share similiar toxicity spread, with LLaMa 8B's top 50 toxic generations have greater score when compared to that of Mixtral 7B.

### Regard

Regard socre is the distribution that a language model would make a positive or negative generation regarding a certain demographic.

In my experiment, I compare language models' prediction of American actors against American actresses and calculate their difference in the regard score. So if it has positive value, the language model is more likely to make certain remakes males than females, and vice versa.

One thing to note is that all models are more likely to make postiive responses regarding females and neutral responses regarding males.

Upon closer inspection, Phi-2 and LLaMa Pro 8B share similiar trend on their generation attitude preference. They both prefer generating neutral response when the topic is male, which has the most significant change. When the topic is female, the models are more likely to generate positive responses. However, in some fewer occasions, both models are more likely to generate negative responses. For Mixtral 7B, the most significant change in generation polarisation is postivie, where the model is more likely to make for female. In contrast, the model is more likely to generate neutral response, and in fewer occasions, negative responses for male.

Phi-2 is the model that has most signification polarisation gap, where most of them eclipse gaps of same attitude compared to the other models. Mixtral 7B has the least signifant gaps compared to other models, making it the least polarising model among them.

### Honest

Honest measure the frequency of hurtful sentence completion by collecting multiple completions of a prompt from a model and aggregate its appearance according to the demographic group which prompt's topic is based on. However, Mixtral 7B and Phi-2 are unable to generate multiple completions of a single prompt due to its API being incompatible with HuggingFace's *num_return_sequences* setting. As a result, I am only able to measure the honest score of LLaMa Pro 8B and make observation of LLaMa Pro 8B based on my limited data.

Evaluate offer two sets of datatest to evaluate the honest score of English language models. One is comparing its sentence completion regarding Queer and Nonqueer, and the other is comparing its sentence completion regarding female and male. For both datasets, LLaMa Pro 8B provides less hurtful sentence completion for protected class (queer and female) than previlleged class (nonqueer and female). The honest score ratio (Protected/Previllege) of LLaMa Pro 8B in case of female and male is much lower than it in the case of queer and nonqueer, meaning it may mean that it has less knowledge of queer and make more biased completion as a result.

### Overall

Phi-2 seem to be overfitting regarding its goal to eliminate bias. It is indeed capable to achieve the lowest toxicity across 200 prompts among all models when measuring the toxicity. However, its drastic performance in regard score difference should be noted. Among the all models, Phi-2 is much more likely to provide a neutral response for male and much more likely to provide a positive as well as a negative respone for female. As a result, it is much more reserved to comment on male individuals while prodcing polarising comments on female individuals. 

Mixtral 7B can be seen to outperform LLaMa Pro 8B in terms of bias elimination. Mixtral 7B perform better in case of toxicity than LLaMa Pro 8B. In case of regard, despite Mixtral 7B prefer producing positive remark for female while LLaMa Pro 8B prefer producing neutral remark for male, the extent of such preference is smaller than that of LLaMa Pro 8B. 

## Comment

### Reliability

- Throughout my experiments, I have not encountered any problems with the framework.

### Generalizability

- Since the framework takes text generated by language model as input to evaluate the bias presents in those generated text, it can be used in language models pipelines other than HuggingFace. It makes it a very flexible framework.
- Technically, you can use prompts sourced from any text datasets to generate text with a language model to evaluate its bias. However, bias metrics work best with prompts from datasets that are engineered to thoroughly test the model. 
    - **Toxicity** metrics are paried with datasets with toxicity prompts
    - **Regard** metrics are paired with **BOLD** datasets
    - **Honest** metrics are paired with **Honest** datasets
- **Honest** metrics requires language model to be able to provide multiple sentence completions from a single prompt. However, there are language models that are unable to achieve such taskes (e.g. Mixtralai's Mixtral 7B, and Microsoft's Phi-2), making the metrics much less accessible for language model particinants to benckmark their models.


### Guidance

- There is a Jupyter Notebook tutorial, providing step-by-step guidance of loading metrics datasets and language models from HuggingFace, using model to generate texts with prompt and evaluate bias present within those generations.
- There are also information regarding each metrics on https://huggingface.co/evaluate-measurement. Each detailing the expected format of its inputs, the format of its output and several indiviual examples.

### Robustness

- The framework can be regarded as a complete package. It means that it can be integrated into any language model workflows.
- All Bias metrics only access the *toxic* aspect of the language models, rather than examine the bias present in the model 
- Both **Regard** and **Honest** have fairness focused on model's generation difference regarding different demographical groups, which may be detrimental  if the language model partitioners' goal is to reduce the amount of toxicity against demographic. Such limitation is less severe for **Honest** due to also providing quantifiable measures.
    - But the requirements to perform a Honest measure means a fraction of interested models are unmeasurable, leading to limited data and lackluster observation for the language model partitioners. (As we can see in my experiment.)
- The framework provided limited bias metricses. All of the bias metrics offered are based on evaluating the text generated from prompts, which lack bias metrics based on correlation within model's word embedding (e.g. WEAT and WEFAT). 
    - As bias metrics of the framework are usually paired with certain prompt datasets, the option the framework provided to evaluate bias in a language model is very limited.