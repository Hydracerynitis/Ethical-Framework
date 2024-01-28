# A preliminary analysis of tabular datasets

Here contains preliminary analysis of tabular datasets I will use when evaluating tabular bias evaluation framework. The goal of the analysis is to grasp a basic understanding of the characteristics of the datatsets (and their bias)

The metric I would use in this preliminary analysis will be:
- Number of instances in the dataset, and number of features in the dataset
- The distribution of labels within the dataset, whether the dataset skews towards positive or negative
- The distribution of protected variables within the datset, whether the dataset  skews towards protected or privilleged
- The distribution of labels given certain procted variables values, which labels correlates more with protected variables groups.

The common protected variables I choose for the research is gender/sex