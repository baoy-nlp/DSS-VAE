## Requirements
- PyTorch 0.4 +
- nltk
- tensorboardX
- Numpy
- PyYAML
- pickle

## Data Preprocess

- [Something]: You need to set it up according to your own situation.
- tokenize   
    ```js
    python preprocess/my_tokenize.py --raw_file [raw_file] --token_file [token_out]
    ```
- prepare raw dataset
    - convert <Constituency Tree> to <Sentence, Linearized Tree> for SyntaxVAE task. 
    [The data needs to be parsed first with an external constituency parser.]   
    ```js
    python preprocess/prepare_raw_set --tree_file [tree_file] --out_file [raw_out] --mode [s2b/s2t/s2s]
    ```       
    - convert <src_file>,<tgt_file> to <src_file,tgt_file> for NAG task.
    ```js
    python preprocess/prepare_raw_set --src_file [src_file] --tgt_file [tgt_file] --out_file [raw_out] --mode [SyntaxVAE/NAG]
    ```
- generate dataset and vocabulary.   
    ```js
    python structs/generate_dataset.py --train_file [train raw_out] -dev_file [dev raw_out] [--test_file [test raw_out] option] --out_dir [output dir] --max_src_vocab [number/-1] --max_src_len [length/-1] --max_tgt_len [length/-1] --train_size [number/-1] --mode [SyntaxVAE/NAG]
    ```
- After Pre-Process, you get the prepared data set as same as:
  ```dir
  data_dir
     +--train.bin
     +--test.bin    
     +--dev.bin
     +──vocab.bin
  ```
  
## Training
```js
python main.py --base_config [data_base_config.yaml] --model_config [model_configs.yaml] --mode train --exp_name [setting-config-name]
```

## Evaluation

## Metric
* Paraphrase
  - Origin BLEU
  - Target BLEU
* Translation
  - Multi Reference BLEU

## Experiments Result


