due to the upload file size limitation, the pairs_new_cpd file in data was splitted into two files

Environment setup：

1. conda instructions
   
   conda env create -f molopt_environment.yaml

   conda activate molopt

2. pip instructions

   pip install -r requirements.txt

Usage examples:

1. First, you can pretrain this model

   python 1_Pretraining.py

2. Then, perform transferlearning training

   python 2_TransferLearning.py

3. generating new optimization strategies

   python 3_Strategies.py
