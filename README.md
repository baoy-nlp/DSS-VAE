Implementation of DSS-VAE: **Generating Sentences from Disentangled Syntactic and Semantic Spaces** in ACL-2019.

## Environment requirements
- PyTorch 0.4 +
- nltk
- tensorboardX
- Numpy
- PyYAML
- pickle

## Data Preparation
Pre: you may need use a constituency parser [ZPar](https://sourceforge.net/projects/zpar/files/0.7.5/zpar-0.7.5.tar.gz/download) for obtaining the constituency parse tree of a sentence.

There are total THREE steps for preprocessing:

1. tokenization
``` shell
python dss_vae/preprocess/my_tokenize.py --raw_file [raw_file_path] --token_file [token_out_path] --for_parse
```
2. parsing 
``` shell
Please refer to ZPar, a easy-to-use constituency parser [ZPar](https://sourceforge.net/projects/zpar/files/0.7.5/zpar-0.7.5.tar.gz/download), for obtaining the constituency parse tree of a sentence.
```
3. build the dataset
- Convert <Constituency Tree> to <Sentence, Linearized Tree>
```shell
python dss_vae/preprocess/tree_linearization.py --tree_file [tree_file_path] --out_file [tree_out_path] --mode s2b
```
- Generate dataset and vocabulary
```shell
python dss_vae/structs/generate_dataset.py --train_file [<Sentence,LinearTree> file] --dev_file [<Sentence,LinearTree> file] --test_file [<Sentence,LinearTree> file] --tgt_dir [output_dir] --max_src_vocab 30000 --max_src_len 30 --max_tgt_len 90 --train_size 100000
```
After Pre-Process, the prepared data directory structure is as follows: 
```
+-- Target Dir
|   +-- train.bin
|   +-- test.bin
|   +-- dev.bin
|   +-- vocab.bin
```
## Training
We can set all the hyper-parametes in the file of config.yaml, and train the model or its variants with the following command:
```shell
python main.py --config_files [config.yaml] --mode train_vae --exp_name [exp_name]
```
Some examples of config.yaml are provided in the directory of CONFIGS.


## Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follow.
```BibTeX
@inproceedings{bao-etal-2019-generating,
    title = "Generating Sentences from Disentangled Syntactic and Semantic Spaces",
    author = "Bao, Yu  and
      Zhou, Hao  and
      Huang, Shujian  and
      Li, Lei  and
      Mou, Lili  and
      Vechtomova, Olga  and
      Dai, Xin-yu  and
      Chen, Jiajun",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1602",
    doi = "10.18653/v1/P19-1602",
    pages = "6008--6019",
}
```