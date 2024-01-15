# Fairpy (LLM)

## Experiment
I have tried using Python 3.11 and 3.10 to set up the toolkit. During trouble shooting I switch to version 3.9 to satisfy its dependencies requirements. Considering three out of four experiments in *demo.py* failed to replicate the result of the toolkit repository, I forfeit to continuing experiment with this toolkit

## Comment

### Stability/Reliability

- There is no install information on the github repository and published paper. This toolkit is not published to a Python package. In order to utilize the toolkit, you need to download the whole repository, extract *BIasDetection* and *BiasMitigation* modules and put it besides your project files.
    - The toolkit is highly dependent on deprecated feature of its dependencies, thus it is not compatible with latest version of its dependency.
    - This result in me mistakely installing another bias detection toolkit with the same name (FairPy)
    - The toolkit is dependent on **transformer**, **torch**, **scikit-learn** ( with **scipy** ), **tensorflow** (with **cuda** and **keras**) and **nltk** packages as mentioned in Github repository. However, these are not mentioned in the *requirements.txt* of the toolkit
        - Despite the Github repository page indicating the toolkit is compatible to Python 3.10, you are unable to satisfy its dependency requirements due to **tensorflow** version 2.7.0 is not available for Python 3.10
        - However, you still need to install **pandas**, **seaborn**, **pattern3**, **ipywidgets**, **honest**, **plotly**, **datasets**, **comet_ml**, **pytorch_lightning** packages if you start with a fresh virtual environment
        - There is also a indentation error in **pattern3** package. Because it is caused by the empty line in try clause, it may be a compatible problem with Python 3.11, 3.10 and 3.9
    - You need to use **nltk** module to downlaod *vader_lexicon* datasets to evaluate WeatScore against BERT model as specified in *requirements.txt*
- The toolkit assume that the python file utilizing the interface it provide is with the same diretory with *BiasDetection* and *BiasMitigation* folder
    - Even withnin the toolkit, the interaction between each files are coded based on such assumption. As a result, you can not isolate the toolkit modules with a folder.
- The toolkit also is not compatible with Jupyter Notebook, as it try to invoke ipython_kernel.py, resulting in *SystemExit*.
- Here's several bugs/problems I encountered during my research:
    - Evaluating the **WeatScore** will result in error due to using a deprecated (and removed) numpy attribute *int*. You need to change it to *int_* to proceed
    - The CausalLMBiasMitigation algorithm breaks due to **ValueError: too many values to unpack (expected 3)**
- When I test the toolkit with the code provided in its Github page, three of the four experiments encounter problems within the toolkits. The exact problem is recorded in *demo.py* file

### Generality

- From what I can understand with the toolkit's code:
    - The toolkit is capable of taking custom language model and its tokenizer as input to evaluate its bias and help mitigate those bias.
    - However, it seems that the toolkit do not accept custom datasets for its bias evaluation and mitigation. It will use data that is builtin within itself to evaluate bias and train a bias-aware model

### Guidance

- There is no external guidance for utilizing the toolkits.
- The toolkit's modular structure is simplistic and straightforward, as available bias metric evaluation and bias mitigation implementation all grouped together under class **BiasDetectionMetrics** and **BiasMitigationMethods**.
- The result of the evaluation methods consists of disconnected jargon. It is diffcult for new partitioner to understand and interpret the result.

## Advantage and Limitation

- Relying on past technology without any forwardcapability, it is very difficulty to integrate this framework into modern day language model workflow. It's incompatibility with Jupyter Notebook is more concerning, since it is the standard way of communication within the machine learning field.
- However, it is the currently only framework that provide language model bias metrics like *WEAT*, and provide bias mitigation methods to fine tune their models.