# CRFM HELM

(CRFM) HELM is a large scale automatic benchmark framework with its operation mainly focused on console and use a localy hosted webpage to deliver benchmark result. It supports 48 scenarios (datasets) to benchmark large language model (LLM) with core scenarios representing possible downstream applications of LLM, and yargeted evaluations designed to evaluate LLMs' skills and risk. In the paper introducing the framework, it has benchmarked 36 state-of-art LLM. It also offers some of those models for users to set up their own benchmark experiment.

## Experiment
I am able to set up the framework with Python 3.8 at Linux server. However, I am only able to run some of its scenario with example models **openai/gpt2** and **bigcode/santacoder** and only get part of metrics that it could produce. In addition, all of the metrics I am able to retrieve from HELM are not related to any topics like bias and fairness. As a result, I choose to forfeit setting up experiment with it until I have the time to figure out the problems in future.

The experiment I plan to set up is to benchmarking models on HuggingFace and models that has been assessed in the paper introducing HELM for comparison. The senario I plan to run are specified in *bias_toxic_template.conf* where I benchmark models in scenarios where bias and toxicity metrics are measured and compared.

## Comment

### Reliability

- During my experience with the HELM, I have encountered problems with HELM that reuslting in:
    - Some of my scenarios specified in my run_spec.conf does not executed during benchmarking
    - Except exact match metrics (and perturbation variation for robustness and fairness), all metrics are unavailable in the final visualised report.
- I was unable to trouble shoot these problems in the end because:
    - The logs of the framework are published on the console due to it being an console-focused framework
    - The logs are filled with warning message like *Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation* making it unreadable.
- However, those problems I encountered may be attributed to having older version of CUDA driver than HELM expected (11.6 rather than 12.0). 

### Generalizability

- Currently, the framework only accepting custom models that are published on Huggingface for benchmarking. It does not accept custom models in local storage.
    - However, since HELM is ruinng with Python 3.8 and using an old version of **Transformers** package. Models that built on newer version of **Transformers** are not compatible with HELM (e.g. **microsoft/phi-2**)
- Currently, HELM does not support custom scenario for benchmarking, but it is a top priority issue for future development. 
- HELM sctrictly can only run with Python version 3.8 and (presumely) CUDA driver version 12.0, on a Linux system that have large amount of memory for storage.
    - The framework will output each individual run's result to a directory with names containing ":", making it incompatible with Windows environemnt. This also make HELM's result untransferrable to other Windows device through source control.
        - As a result, I do not save my remaining findings on GitHub repository to prevent my repository from being unable to update due to it will be stuck at "git pull" stage.
    - During benchmark, it will download datasets and store them in the .cache folder in user directory. Depending the numbers of different scenarios it has run, the .cache folder can take out up to 190GB memory in your system.
    - The incompatiblity of Python version and CUDA version is explained in *Stability/Reliabity* sections
- Most LLM that HELM provide open access (or limited access) to are deployed on together and require a together API in credential.conf in the production environment to make generations during benchmarking.

### Guidance

- There is a Documentations site for [HELM](https://crfm-helm.readthedocs.io/en/latest/tutorial/), including installation information and explaination of basic operation of the framework.
- The authros of HELM also host the [report](https://crfm.stanford.edu/helm/v0.2.2/?) of their paper's results online for access. It contian information of availability of supported LLM and descriptions and evaluated metrics of each scenarios. It can be used as a guidance for expected results from the framework.
- However, there is little information about the accepted format of run_spec.conf file, which are used to specify benchmark tasks. The only reference point is the [run_spec.conf](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/presentation/run_specs.conf) file that the paper used for its experiment.
    - Designing a run_spec.conf file that can run large amount of scenarios for multiple models are tedious, since for each model and each scenarios, you have to write a line to indicate such task in the file
        - This can be solved by using "model=text" to indicate running all text models with those scenarios and use --models-to-run flag to indicate the framework that only models specified in the flag are used in the benchmark. However, such methods do not support custom models from HuggingFace.
- It also has a lightweight variation called HELM Lite. Currently there is no documentation about utilizing it, but it is a top priority issue for future development.
 
### Robustness

- HELM has a capacibility to support a large-scale benchmarking of current available LLM as shown in its paper.Its in-depth selection of scenarios is able to support thourough evaluation of effectiveness of LLMs. It also help aggregate API accesses of state-of-art LLMs into one place for an efficiet industry-wide accessment.
    - When custom LLM and custom scenario feature are developed, it can provide greate reference points for evaluating newly developed LLM as well as initiate an industry-wide evaluation when a new accessment scenario is developed.
- As HELM requires a Linux system with large storage space, users usually requires to gain access a dedicated server to run benchmark with it. It can be a potential barrier preventing people from engaging a large-scale benchmarking with the framework. 
- However, most of its metrics are irrelevant to bias and fairness, making it a less effective framework as a bias evaluation framework.
    - The metrics that are relevant to bias and fairness are: pertubated accuracy regarding demographic  (pertubated accuracy for short), demographic representation, stereotypical associations, and toxicity fraction.
    - However, pertubated accuracy is not connected with the other three metrics. According to the paper's result, most of the scenarios where pertubated accuracy is recorded do not calculate the other three metrics.
- Bias and Fairness relevant metrics themselver are also weak in their definition as noted in the [paper](https://arxiv.org/pdf/2211.09110.pdf). Demographic representation and stereotypical associations are count-based metrics where they count occurance of demographic related words and compare them with totla occurance or their correlation with certain words. Such count-based metrics is rigid and sometimes may lead to undesirable values. 
