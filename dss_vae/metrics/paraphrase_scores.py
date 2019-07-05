# coding=utf-8

import os
import re


def extract_score_file_list(file_dir):
    L = []
    for root, _, files in os.walk(file_dir):
        for file in files:
            if file.endswith(".score"):
                L.append(os.path.join(root, file))
    return L


def extract_score_to_tgt(src_file):
    # "ori bleu:{}, tgt bleu:{} !".
    score_file_list = extract_score_file_list(src_file)
    dev_score = []
    test_score = []

    def score_item(line):
        items = line.strip()
        scores = re.findall(r"\d+\.?\d*", items)
        return "\t".join(scores)

    for file in score_file_list:
        with open(file, "r") as f:
            dev_line = f.readline()
            test_line = f.readline()
            dev_score.append(score_item(dev_line))
            test_score.append(score_item(test_line))

    with open(src_file + "para.dev", "w") as f:
        for item in dev_score:
            f.write(item)
            f.write("\n")

    with open(src_file + "para.test", "w") as f:
        for item in test_score:
            f.write(item)
            f.write("\n")


import sys

extract_score_to_tgt(sys.argv[1])
