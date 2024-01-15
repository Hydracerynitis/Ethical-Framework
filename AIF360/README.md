# AI Fariness 360

## Experiment

In my previous research, I use the toolkit to benchmark its bias mitgation implementations against three datasets I gathered in Kaggle. I use Python 3.11 through out the experiment.

## Comment

### Generality

- Both bias detection and bias mitigation feature only accept the toolkit's only data structure **StandardDataset**
- However, the toolkit provide conversion methods to convert **DataFrame** class to **StandardDataset**
- The conversion would handle replacing categorical values into discret integer values, removing rows with missing values, drop redundant feature (specified in the parameter), identify favourable protected attributes and labels and transform them to 0 and 1.
- This also means that the toolkit is limited to binary classifcation problems with bianry previllege groups (previlleged and unprevilleged)
- **StandardDataset** provides method to split itself randomly into training sets and testing sets. It also provides method to return a subset of itself based on a list of index.
- However, **StandardDataset** lack the flexibility and modifiability the **Dataframe** class provides. 
    - For example, you can split a **Dataframe** into five equal size folds, and contact four folds to form the training sets for cross validation. This can not be performed with **StandardDataset** because it does not provide interfaces to combine different instances of **StandardDataset** together
- Bias metrics evaluation and bias mitigation implements can accpet custom Machine Learning model, to evaluate its bias metrics and mitigate its bias (except for in-process bias mitgation methods due to intefering training process directly)

### Stability/Reliability

- Here's several bugs/problems I encountered during my prior research:
    - The conversion method from **Dataframe** to **StandardDataset** can break when given a dataframe instances that have 130 columns.
    - The bias mitigation implementation of **Reweight** and **RejectOptionClassifier** have problems when running.

### Guidance

- The README.md of its github repository provide detailed information on its installments. It also provide several installment options with different supported feature and dependencies
- There are plenetiful learning resources in its Github repository for new paritioners to get familiar with the toolkit.
- The learning resources include:
    - A demo notebooks with each bias mitgation methods
    - Several tutorial of using the toolkits
- Its methods and interfaces are well documented for the purpose of the parameter, and it has provided detail documentation on the website https://aif360.readthedocs.io/en/stable/index.html
- It has provided mathematical definition of each supported bias metrics on the documentation website
- It also provides example datasets for new partitioners to experiment with

## Advantage and Limitation

- The framework use **StandardDataset** class as its data struture provides convenience like automating basic data pre-process tasks like transforming categorical features and categorize privillege groups. 
    - However, it is less flexible than panda's **Dataframe** class to query its content and perform various transformation operations, making it harder to debug if encountered problems when using the framework.
    - And if encountered problems during conversion between **Dataframe** and **StandardDataset**, if may means that you can not perform bias evaluation and migitation with this framework
- It provide evluations of plentiful bias metrics. However, most of them have definition based on predicted outcomes and actual outcomes or predicted probability and actual outcome, lacking those definition based on predicited outcome, similarity measures and casual reasoning 
- It provides implementations of state-of-art bias mitigations methods, that can be very convenient for any machine learning workflow to improve their fairness. However, there is significantly more implementations of in-processing migitation methods than those of pre-processing and post-processing ones.
    - Since implementations of in-processing migitation methods present a new machine learning training processs rather than modify an exisiting ones, they can not be able to integrated with exisitng workflows and have to actively compete with them
