# Fairness Indicator

Fairness Indicator is a bias evaluation framework that built upon Tensorflow Model Analysis. It position itself for tensorflow machine learning model workflow integration.

## Experiment


## Comment

### Stability/Reliability

- The Fairness Indicator is built upon Tensorflow Model Analysis, which is only available in Python 3.8 and 3.9. This installation preequisite is not mention on either the github page or tutorials.
- There are several instances of installing fairness indicators:
    - Installing the **fairness-indicators** of version 0.43.0 and 0.42.0 on a newly created virtual environment (both on Python 3.8 and 3.9) will result in dependency conflict with its own depnedency **apache-beam**. 
        - This is due to **fairness-indicators**'s dependency **tensorflow-model-analysis** (0.44.0) requires package **attrs** of version >=19.3.0 and <0.22, while **apache-beam**'s dependency **jsonschema** requires package **attrs** of version >=22.2.0
- I have encountered a dependecy conflict between fairness-indicators and scikit-learn 

### Generality



### Guidance

- It provides Colab notebooks tutorial to provide guidance for new partitioners. However, the notebook is run on Python version 3.10. As Fairness-Indicators's preequisite is Python version 3.8 and 3.9, those notbooks tutorial are not currently working.

## Advantage and Limitation

