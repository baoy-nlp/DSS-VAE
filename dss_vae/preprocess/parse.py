from six import print_
from zpar import ZPar

# use the zpar wrapper as a context manager
with ZPar('english-models') as z:
    # get the parser and the dependency parser models
    tagger = z.get_tagger()
    depparser = z.get_depparser()
    parser = z.get_parser()

    # tag a sentence
    tagged_sent = tagger.tag_sentence("I am going to the market.")
    print_(tagged_sent)

    # tag an already tokenized sentence
    tagged_sent = tagger.tag_sentence("Do n't you want to come with me to the market ?", tokenize=False)
    print_(tagged_sent)

    # get the dependency parse of an already tagged sentence
    dep_parsed_sent = depparser.dep_parse_tagged_sentence("I/PRP am/VBP going/VBG to/TO the/DT market/NN ./.")
    print_(dep_parsed_sent)

    # get the dependency parse of an already tokenized sentence
    dep_parsed_sent = depparser.dep_parse_sentence("Do n't you want to come with me to the market ?", tokenize=False)
    print_(dep_parsed_sent)

    # get the dependency parse of an already tokenized sentence
    # and include lemma information (assuming you have NLTK as well
    # as its WordNet corpus installed)
    dep_parsed_sent = depparser.dep_parse_sentence("Do n't you want to come with me to the market ?", tokenize=False,
                                                   with_lemmas=True)
    print_(dep_parsed_sent)



