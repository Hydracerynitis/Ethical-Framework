# Fairpy (LLM)

## Experiment

First use Python 3.11 when setting up the toolkit, later switch to 3.10 when figuring out a compatibility problem. The expeirment formally uses Python 3.10

### Stability/Reliability

- There is no install information on the github repository and published paper. This toolkit is not published to a Python package. In order to utilize the toolkit, you need to download the whole repository, extract *BIasDetection* and *BiasMitigation* modules and put it besides your project files.
    - This result in me mistakely installing another bias detection toolkit with the same name (FairPy)
    - The toolkit is dependent on **transformer**, **torch**, **scikit-learn** ( with **scipy** ), **tensorflow** (with **cuda** and **keras**) and **nltk** packages as mentioned in Github repository. However, these are not mentioned in the *requirements.txt* of the toolkit
        - However, you still need to install **pandas**, **seaborn**, **pattern3**, **ipywidgets**, **honest**, **plotly**, **datasets**, **comet_ml**, **pytorch_lightning** packages if you start with a fresh virtual environment
        - The toolkit is not compatible with the latest version of **transformers** package, rather depend on version 4.21.0 as specified in the Github Repository
        - There is also a indentation error in **pattern3** package. Because it is caused by the empty line in try clause, it may be a compatible problem with Python 3.11 and 3.10
    - You need to use **nltk** module to downlaod *vader_lexicon* datasets to evaluate WeatScore against BERT model as specified in *requirements.txt*
- The toolkit assume that the python file utilizing the interface it provide is with the same diretory with *BiasDetection* and *BiasMitigation* folder
    - Even withnin the toolkit, the interaction between each files are coded based on such assumption. As a result, you can not isolate the toolkit modules with a folder.
- The toolkit also is not compatible with Jupyter Notebook, as it try to invoke ipython_kernel.py, resulting in *SystemExit*.
- Here's several bugs/problems I encountered during my research:
    - Evaluating the **WeatScore** will result in error due to using a deprecated (and removed) numpy attribute *int*. You need to change it to *int_* to proceed
    - The CausalLMBiasMitigation algorithm breaks due to **ValueError: too many values to unpack (expected 3)**
