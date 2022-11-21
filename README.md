## The steps to run are as follows:

1. Data processing

    1.1 Obtain dataset from https://github.com/OanaMariaCamburu/e-SNLI;
        
    1.2 Extract rationlaes, which can refer to https://github.com/SCUT-CCNL/Expl-NLI;

2. Processing the knowledge graph (ConceptNet)

    2.1 Using ./process/extract_cpnet.py to extract concepts from ConceptNet;
    
    2.2 Us ./process/graph_construction.py to construct the corresponding graph of ConceptNet;
    
    2.3 Using ./process/snli_ground_concepts_simple.py to extract the concepts in the e-SNLI dataset;
    
    2.4 Using ./process/snli-diff_find_neighbors.py to construct the subgraph of each sample in e-SNLI;
    
    2.5 Using ./process/snli_filter_triple.py to obtain the final subgraph by the rule introduced in our paper;
    
3. Model train
    3.1 Download the corresponding files of gpt2 model from "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-XXX;
    
    3.2 Train the KxNLI model using snli_main.py with default parameters;
    
4. Model inference
    4.1 Using ./train_and_eval/snli_inference.py for model inference with default parameters;
    
    4.2 Using ./train_and_eval/eval_NLE.py to evaluate the generated NLEs;
    
    4.3 Using ./train_and_eval/snli_RoBERTa.py to measure the faithfulness of the generated NLEs;
    
5. Transfer performance: replace the SNLI dataset with the MultiNLI dataset, and then perform the above steps;
6. Execute the idea of this article on the vanilla seq2seq model
    6.1 Prepare the required data and save it in the ./vanilla_seq2seq/data directory
    
    6.2 Using the TransE model to encode concepts in subgraphs, refer to OpenKE (https://github.com/thunlp/OpenKE)
    
    6.3 Using ./vanilla_seq2seq/train.py for model train;
    
    6.4 Use ./vanilla_seq2seq/infer.py for model inference;

# Acknowledgment:

This repo is built upon the following works:

1.  LIREx: Augmenting Language Inference with Relevant Explanations.
https://github.com/zhaoxy92/LIREx

2. Language Generation with Multi-hop Reasoning on Commonsense Knowledge Graph
https://github.com/cdjhz/multigen

Many thanks to the authors and developers!
