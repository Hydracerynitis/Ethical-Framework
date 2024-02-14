# CRFM HELM

(CRFM) HELM is a large-scale automatic benchmark framework whose operation is mainly focused on the terminal. It uses a locally hosted webpage to deliver benchmark results. It supports 48 scenarios (datasets) to benchmark large language models (LLM) with core scenarios representing possible downstream applications of LLM and targeted evaluations designed to evaluate LLMs' skills and risk. The paper introducing the framework has benchmarked 36 state-of-the-art LLMs. It also offers some models for users to set up their benchmark experiments.

## Experiment
I can set up the framework with Python 3.8 on a Linux server. However, I can only run some of its scenarios with example models **openai/gpt2** and **bigcode/santacoder** and only get part of the metrics it could produce. In addition, none of the metrics I can retrieve from HELM are related to bias and fairness. As a result, I choose to forgo setting up experiments with it until I have the time to figure out the problems in the future.

The experiment I plan to set up is to benchmark models on HuggingFace and models assessed in the paper introducing HELM for comparison. The scenarios I plan to run are specified in *bias_toxic_template.conf*, where I benchmark models in scenarios where bias and toxicity metrics are measured and compared.

## Comment

### Reliability

- During my experience with the HELM, I have encountered problems with HELM that resulting in:
    - Some of my scenarios specified in my run_spec.conf does not execute during benchmarking
    - Except for exact match metrics (and perturbation variation for robustness and fairness), all metrics are unavailable in the final visualised report.
- I was unable to troubleshoot these problems in the end because:
    - The logs of the framework are published on the console due to it being a console-focused framework
    - The logs are filled with warning message like *Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation* making it unreadable.
- However, those problems I encountered may be attributed to having an older version of the CUDA driver than HELM expected (11.6 rather than 12.0). 

### Generalizability

- Currently, the framework only accepts custom models published on Huggingface for benchmarking. It does not accept custom models in local storage.
    - However, since HELM is running with Python 3.8 and using an old version of the **Transformers** package. Models that built on newer version of **Transformers** are not compatible with HELM (e.g. **microsoft/phi-2**)
- Currently, HELM does not support custom scenarios for benchmarking, but it is a top priority issue for future development. 
- HELM strictly can only run with Python version 3.8 and (presumably) CUDA driver version 12.0 on a Linux system with a large amount of memory for storage.
    - The framework will output each run's result to a directory with names containing ":", making it incompatible with the Windows environment. This problem makes HELM's result utterly incompatible with Windows environments.
        - As a result, I do not save my remaining results on the GitHub repository to prevent my repository from being unable to update because it is stuck at the "git pull" stage.
    - During benchmarking, it will download datasets and store them in the .cache folder in the user directory. Depending on the number of scenarios it has run, the .cache folder can take out up to 190GB of memory in your system.
    - The incompatibility between the Python version and the CUDA version is explained in the *Reliability* sections
- Most LLMs that HELM provides open access (or limited access) are deployed together and require a combined API in credential in the production environment to make generations during benchmarking.

### Guidance

- There is a documentation site for [HELM](https://crfm-helm.readthedocs.io/en/latest/tutorial/), including installation information and an explanation of the basic operation of the framework.
- The authros of HELM also host the [report](https://crfm.stanford.edu/helm/v0.2.2/?) of their paper's results online for access. It contains information on the availability of supported LLM and descriptions and evaluated metrics of each scenario. It can be used as a guidance for expected results from the framework.
- However, there is little information about the accepted format of run_spec.conf file, which is used to specify benchmark tasks. The only reference point is the [run_spec.conf](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/presentation/run_specs.conf) file that the paper used for its experiment.
    - Designing a run_spec.conf file that can run a large number of scenarios for multiple models is tedious since, for each model and each scenario, you have to write a line to indicate such task in the file
        - This can be solved by using "model=text" to indicate running all text models with those scenarios and using the --models-to-run flag to indicate the framework that only models specified in the flag are used in the benchmark. However, such methods do not support custom models from HuggingFace.
- It also has a lightweight variation called HELM Lite. There is no documentation about utilising it, but it is a top priority issue for future development.
 
### Robustness

- HELM can support a large-scale benchmarking of current available LLM, as shown in its paper. Its in-depth selection of scenarios can support a thorough evaluation of LLMs' effectiveness. It also helps aggregate API access to state-of-the-art LLMs into one place for an efficient industry-wide assessment.
    - When custom LLM and custom scenario features are developed, HELM can provide excellent tests to evaluate newly developed LLM and initiate an industry-wide evaluation when a new assessment scenario is developed.
- As HELM requires a Linux system with large storage space, users are usually required to gain access to a dedicated server to run benchmarks with it. It can be a potential barrier preventing people from engaging in large-scale benchmarking with the framework. 
- However, most of its metrics are irrelevant to bias and fairness, making it a less effective bias evaluation framework.
    - The metrics relevant to bias and fairness are perturbed accuracy regarding demographics (perturbed accuracy for short), demographic representation, stereotypical associations, and toxicity fraction.
    - However, perturbed accuracy is not connected with the other three metrics. According to the paper's result, most scenarios where perturbed accuracy is recorded do not calculate the other three metrics.
- Bias and Fairness-relevant metrics themselves are also weak in their definition, as noted in the [paper](https://arxiv.org/pdf/2211.09110.pdf). Demographic representation and stereotypical associations are count-based metrics that count the occurrence of demographic-related words and compare them with their total accuracy or correlation with certain words. Such count-based metrics are rigid and sometimes may lead to undesirable values. 

## How to utilise this subrepo

This subrepo has been used in Linux environments. 

This subrepo will utilise CUDA. My Linux CUDA driver version is 11.6

To use this subrepo in Linux Environment, use Python version 3.8 and [linux_requirements.txt](./linux_requirements.txt) with pip to install the virtual environment. It is mainly run on terminals with command lines. Most of its information can be obtained from its GitHub page and official website.

*bias_toxic_template.conf* are modified from the framework's paper configuration file. It is modified only to include scenarios with toxicity and bias metrics accessed. *format_run_spec.py* contains Python scripts that for each models in *model.txt*, create a modified version of *bias_toxic_template.conf* that only acessed this model. *console_command.txt* records the commonly used HELM commands: running HELM with a configuration file and executing Python scripts.