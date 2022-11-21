## The steps to run are as follows:

1. Data processing
        1.1 Obtain dataset from https://github.com/OanaMariaCamburu/e-SNLI ;
        
        1.2 Extract rationlaes: refer to https://github.com/SCUT-CCNL/Expl-NLI;

2. Processing the knowledge graph (ConceptNet)

    2.1 Using extract_cpnet.py to extract concepts from ConceptNet;
    
    2.2 Us graph_construction.py to construct the corresponding graph of ConceptNet;
    
    2.3 Using snli_ground_concepts_simple.py to extract the concepts in the e-SNLI dataset;
    
    2.4 Using snli-diff_find_neighbors.py to construct the subgraph of each sample in e-SNLI;
    
    2.5 Using snli_filter_triple.py to obtain the final subgraph by the rule introduced in our paper;
    
3. Train the model
    3.1 Download the corresponding version of gpt2 models. "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-XXX.
    
    3.2 Run snli_main.py to train the model, and the training parameters all use the default values in the code;
    
4. Model reasoning
    4.1 Run snli_inference.py to do model inference, and the parameters also use the default values in the file;
    
    4.2 Run eval_NLE.py to evaluate the produced NLE;
    
    4.3 Run snli_RoBERTa.py to assess the adequacy and completeness of the NLE;
    
5. Migration performance: replace the SNLI dataset with the MultiNLI dataset, and then perform the above steps;
6. Execute the idea of this article on the vanilla seq2seq model
    6.1 Prepare the required data and save it in the ./data directory
    
    6.2 Use the TransE model to encode concepts in subgraphs, refer to: openke
    
    6.3 Use train.py to train the model
    
    6.4 Use infer.py to reason;

# Acknowledgment:

This repo is built upon the following works:

1.  LIREx: Augmenting Language Inference with Relevant Explanations.
https://github.com/zhaoxy92/LIREx

2. Language Generation with Multi-hop Reasoning on Commonsense Knowledge Graph
https://github.com/cdjhz/multigen

Many thanks to the authors and developers!
