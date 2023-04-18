# Files used to train the models 
### preprocessing.py
- Code to preprocess the cti texts and replace special terms and substitute them 
- preprocessing_selected.py uses only the best combination of preprocessing
### train_t5_base_config.py
- usage python train_t5_base_config.py T5_CTI.yaml
- the basic train script 
### train_t5_base_with_pipe_selected.py
- the final train script 
### utils_re_pipe.py
- utils to calculate metrics for the model 
- strict means no levenshtein distance used 