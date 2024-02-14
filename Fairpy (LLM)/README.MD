# Fairpy (LLM)

Fairpy (designed for language models, not to be confused with the framework with the same name targeting tabular models) is a framework that has gathered state-of-the-art bias evaluation methods for language models and bias mitigation methods. 

## Experiment
I have tried using Python 3.11 and 3.10 to set up the toolkit. Ultimately, I switched to version 3.9 to satisfy its dependency requirements. Considering three out of four experiments in *demo.py* failed to replicate the result of the toolkit repository, I forfeit setting up an experiment with this toolkit.

The experiment I plan to do is to run WEAT, SEAT and StereoSet tests on models popular on HuggingFace and GPT-2 as a comparison. Then, I will use the WEAT, SEAT and StereoSet scores to analyse which exhibits the most bias in the models' word embeddings.

## Comment

### Reliability

- There is no install information on the GitHub repository and published paper. This toolkit is not published to a Python package. To utilise the toolkit, you need to download the whole repository, extract *BIasDetection* and *BiasMitigation* modules and put them beside your project files.
    - The toolkit is highly dependent on the deprecated features of its dependencies. Thus, it is not compatible with the latest version of its dependency.
    - This resulted in me mistakenly installing another bias detection toolkit with the same name (FairPy)
    - The toolkit is dependent on **transformer**, **torch**, **scikit-learn** ( with **scipy** ), **tensorflow** (with **cuda** and **keras**) and **nltk** packages as mentioned in Github repository. However, these are not mentioned in the *requirements.txt* of the toolkit
        - Despite the Github repository page indicating the toolkit is compatible with Python 3.10, you are unable to satisfy its dependency requirements due to **tensorflow** version 2.7.0 is not available for Python 3.10
        - However, you still need to install **pandas**, **seaborn**, **pattern3**, **ipywidgets**, **honest**, **plotly**, **datasets**, **comet_ml**, **pytorch_lightning** packages if you start with a fresh virtual environment
        - There is also an indentation error in the **pattern3** package. Because it is caused by the empty line in the try clause, it may be a compatible problem with Python 3.11, 3.10 and 3.9
    - You need to use **nltk** module to download *vader_lexicon* datasets to evaluate WeatScore against BERT model as specified in *requirements.txt*
- The toolkit assumes that the python file utilising the interface it provides is with the same directory with the *BiasDetection* and *BiasMitigation* folder
    - Even within the toolkit, the interaction between each file is coded based on such assumption. As a result, you can not isolate the toolkit modules with a folder.
- The toolkit is also incompatible with Jupyter Notebook, as it tries to invoke ipython_kernel.py, resulting in *SystemExit*.
- Here are several bugs/problems I encountered during my research:
    - Evaluating the **WeatScore** will result in an error due to using a deprecated (and removed) numpy attribute *int*. You need to change it to *int_* to proceed
    - The CausalLMBiasMitigation algorithm breaks due to **ValueError: too many values to unpack (expected 3)**
- When I tested the toolkit with the code provided on its GitHub page, three of the four experiments encountered problems within the toolkits. The exact problem is recorded in the *demo.py* file

### Generalizability

- From what I can understand with the toolkit's code:
    - The toolkit is capable of taking a custom language model and its tokeniser as input to evaluate its bias and help mitigate that bias.
    - However, the toolkit does not seem to accept custom datasets for its bias evaluation and mitigation. It will use data that is built within itself to evaluate bias and train a bias-aware model

### Guidance

- There is no external guidance for utilising the toolkits.
- The toolkit's modular structure is simplistic and straightforward, as available bias metric evaluation and mitigation implementation are all grouped under class **BiasDetectionMetrics** and **BiasMitigationMethods**.
- The result of the evaluation methods consists of disconnected jargon. It is difficult for new users to understand and interpret the result.

### Robustness

- It is suspected that the framework is incompatible with the Windows environment. However, I have no time to reinvestigate this framework at the end of the survey's timeframe. Future work can be done to reinvestigate it in Linux environments.
- Relying on past technology without any forward capability, it is very difficult to integrate this framework into modern-day language model workflow. Its incompatibility with the Jupyter Notebook is more concerning since it is the standard way of communication within the machine learning field.
- However, it is currently the only framework that provides language model bias metrics like *WEAT* and bias mitigation methods to fine-tune their models.

## How to utilise this subrepo

This subrepo has been used in both Windows environments. 

To use this subrepo in the Window Environment, use Python version 3.9 and [window_requirements.txt](./window_requirements.txt) with pip to install the virtual environment.

*demo.py* and *demo.ipynb* are used when I try to utilise this framework.