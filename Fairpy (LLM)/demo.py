import nltk
nltk.download('vader_lexicon')
from BiasDetection import BiasDetectionMetrics
from BiasMitigation import BiasMitigationMethods

# maskedObj = BiasDetectionMetrics.MaskedLMBiasDetection(model_class = 'bert-base-uncased')
# maskedObj.WeatScore(bias_type='health') 
'''
Null hypothesis: no difference between MentalDisease and PhysicalDisease in association to attributes Temporary and Permanent   
Equalities contributed 1/924 to p-value
pval: 0.568
effect size: -0.111
Percentage of p_value <0.05: 0.0
Average E-score: 0
'''

# causalObj = BiasDetectionMetrics.CausalLMBiasDetection(model_class = 'gpt2')
# causalObj.stereoSetScore(bias_type='gender') #require CUDA, require to use new PC
'''
The Evaluation Breaks
The StackTrace:
 File ".\demo.py", line 18, in <module>
    causalObj.stereoSetScore(bias_type='gender')
  File ".\BiasDetection\BiasDetectionMetrics.py", line 129, in stereoSetScore
    stereoset_obj = StereoSetGender(self.model, self.tokenizer, self.device, self.model_class, 'causal', self.MSK, dataset)     
  File ".\BiasDetection\metrics\StereoSetMetric\StereoSetGender.py", line 7, in __init__
    super().__init__(model, tokenizer, device, model_class, model_type, mask_token, dataset)
  File ".\BiasDetection\metrics\StereoSetMetric\StereoSet.py", line 8, in __init__
    self.stereoObj = causalBiasEvaluator(model, device, pretrained_class=model_class, tokenizer=tokenizer, input_file=self.input_file)
  File ".\BiasDetection\metrics\StereoSetMetric\code\eval_generative_models.py", line 64, in __init__
    self.dataloader = dataloader.StereoSet(filename)
  File ".\BiasDetection\metrics\StereoSetMetric\code\dataloader.py", line 115, in __init__
    with open(location, "r") as f:
FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\hrish\\Documents\\Purdue\\Summer 22\\Language Bias\\Bias Detection\\bias-bench-maindata\\StereoSetData\\dev.json'
'''

MaskedMitObj = BiasMitigationMethods.MaskedLMBiasMitigation(model_class='bert-base-uncased')
model, tokenizer = MaskedMitObj.NullSpaceProjection('bert-base-uncased', 'BertForMaskedLM', 'race', train_data='yelp_sm')
'''
The Implementation breaks
The StackTrace:
File ".\demo.py", line 38, in <module>
    model, tokenizer = MaskedMitObj.NullSpaceProjection('bert-base-uncased', 'BertForMaskedLM', 'race', train_data='yelp_sm')   
File ".\BiasMitigation\BiasMitigationMethods.py", line 240, in NullSpaceProjection
    projection_matrix = ComputeProjectionMatrix(model, tokenizer, model_class, dataset, train_data, bias_type)
File ".\BiasMitigation\techniques\NullSpaceProjection\inlp_projection_matrix.py", line 20, in ComputeProjectionMatrix
    data = load_inlp_data('BiasMitigation', bias_type, dataset, seed=seed)
File ".\BiasMitigation\techniques\NullSpaceProjection\inlp.py", line 20, in load_inlp_data
    data = _load_race_data(persistent_dir, dataset)
File ".\BiasMitigation\techniques\NullSpaceProjection\inlp.py", line 155, in _load_race_data
    lines = f.readlines()
UnicodeDecodeError: 'gbk' codec can't decode byte 0x8b in position 3367: illegal multibyte sequence
'''

# CausalMitObj = BiasMitigationMethods.CausalLMBiasMitigation(model_class='gpt2')
# model, tokenizer = CausalMitObj.DropOutDebias('gpt2', 'religion', train_data='yelp_sm')
'''
The Implementation breaks
The StackTrace:
File ".\demo.py", line 22, in <module>
    model, tokenizer = CausalMitObj.DropOutDebias('gpt2', 'religion', train_data='yelp_sm')
File ".\BiasMitigation\BiasMitigationMethods.py", line 114, in DropOutDebias
    model, tokenizer = causalRetrain(model_name_or_path=model_class, output_dir='savedModel/', train_file=train_data, counterfactual_augmentation=bias_type, do_train=True, seed=4,
File ".\BiasMitigation\techniques\LMRetrain\causalLMRetrain.py", line 708, in Retrain
    tokenized_datasets = tokenized_datasets.map(
File ".\.venv\lib\site-packages\datasets\dataset_dict.py", line 855, in map
    {
File ".\.venv\lib\site-packages\datasets\dataset_dict.py", line 856, in <dictcomp>
    k: dataset.map(
File ".\.venv\lib\site-packages\datasets\arrow_dataset.py", line 591, in wrapper
    out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
File ".\.venv\lib\site-packages\datasets\arrow_dataset.py", line 556, in wrapper
    out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
File ".\.venv\lib\site-packages\datasets\arrow_dataset.py", line 3089, in map
    for rank, done, content in Dataset._map_single(**dataset_kwargs):
File ".\.venv\lib\site-packages\datasets\arrow_dataset.py", line 3466, in _map_single
    batch = apply_function_on_filtered_inputs(
File ".\.venv\lib\site-packages\datasets\arrow_dataset.py", line 3345, in apply_function_on_filtered_inputs
    processed_inputs = function(*fn_args, *additional_args, **fn_kwargs)
File ".\BiasMitigation\techniques\LMRetrain\causalLMRetrain.py", line 648, in ternary_counterfactual_augmentation
    r1_word, r2_word, r3_word = augmentation_words
ValueError: too many values to unpack (expected 3)
'''
