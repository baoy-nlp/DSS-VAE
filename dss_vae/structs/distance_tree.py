import re

import nltk
import numpy

word_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR',
             'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
             'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP',
             'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
             'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
currency_tags_words = ['#', '$', 'C$', 'A$']
eclipse = ['*', '*?*', '0', '*T*', '*ICH*', '*U*', '*RNR*', '*EXP*', '*PPA*', '*NOT*']
punctuation_tags = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``']
punctuation_words = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``',
                     '--', ';', '-', '?', '!', '...', '-LCB-',
                     '-RCB-']
deleted_tags = ['TOP', '-NONE-', ',', ':', '``', '\'\'']


def process_arc(label):
    labels = label.split('+')
    new_arc = []
    for sub_labels in labels:
        if sub_labels == 'ADVP':
            sub_labels = 'PRT'
        new_arc.append(sub_labels)
    label = '+'.join(new_arc)
    return label


def process_none(tree):
    if isinstance(tree, nltk.Tree):
        label = tree.label()
        if label == '-NONE-':
            return None
        else:
            tr = []
            for node in tree:
                new_node = process_none(node)
                if new_node is not None:
                    tr.append(new_node)
            if len(tr) == 0:
                return None
            else:
                return nltk.Tree(label, tr)
    else:
        return tree


def build_nltk_tree(depth, arc, tag, sen, arc_dict, tag_dict, stag_dict, stags=None):
    """
    stags are the stanford predicted tags present in the train/valid/test files.
    """
    assert len(sen) > 0
    assert len(depth) == len(sen) - 1, ("%s_%s" % (len(depth), len(sen)))
    if stags:
        assert len(stags) == len(tag)

    if len(sen) == 1:
        tag_list = str(tag_dict[tag[0]]).split('+')
        tag_list.reverse()
        # if stags, put the real stanford pos TAG for the word and leave the
        # unary chain on top.
        if stags is not None:
            assert len(stags) > 0
            tag_list.insert(0, str(stag_dict[stags[0]]))
        word = str(sen[0])
        for t in tag_list:
            word = nltk.Tree(t, [word])
        assert isinstance(word, nltk.Tree)
        return word
    else:
        idx = numpy.argmax(depth)
        node0 = build_nltk_tree(
            depth[:idx], arc[:idx], tag[:idx + 1], sen[:idx + 1],
            arc_dict, tag_dict, stag_dict, stags[:idx + 1] if stags else None)
        node1 = build_nltk_tree(
            depth[idx + 1:], arc[idx + 1:], tag[idx + 1:], sen[idx + 1:],
            arc_dict, tag_dict, stag_dict, stags[idx + 1:] if stags else None)

        if node0.label() != '<empty>' and node1.label() != '<empty>':
            tr = [node0, node1]
        elif node0.label() == '<empty>' and node1.label() != '<empty>':
            tr = [c for c in node0] + [node1]
        elif node0.label() != '<empty>' and node1.label() == '<empty>':
            tr = [node0] + [c for c in node1]
        elif node0.label() == '<empty>' and node1.label() == '<empty>':
            tr = [c for c in node0] + [c for c in node1]

        arc_list = str(arc_dict[arc[idx]]).split('+')
        arc_list.reverse()
        for a in arc_list:
            if isinstance(tr, nltk.Tree):
                tr = [tr]
            tr = nltk.Tree(a, tr)

        return tr


def mrg(tr):
    if isinstance(tr, str):
        return '( %s )' % tr
        # return tr + ' '
    else:
        s = '('
        for subtr in tr:
            s += mrg(subtr) + ' '
        s += ')'
        return s


def get_brackets(tree, start_idx=0, root=False):
    assert isinstance(tree, nltk.Tree)
    label = tree.label()
    label = label.replace('ADVP', 'PRT')

    brackets = set()
    if isinstance(tree[0], nltk.Tree):
        end_idx = start_idx
        for node in tree:
            node_brac, next_idx = get_brackets(node, end_idx)
            brackets.update(node_brac)
            end_idx = next_idx
        if not root:
            brackets.add((start_idx, end_idx, label))
    else:
        end_idx = start_idx + 1

    return brackets, end_idx


def normalize(x):
    return x / (sum(x) + 1e-8)


def tree2list(tree, parent_arc=[]):
    if isinstance(tree, nltk.Tree):
        label = tree.label()
        if isinstance(tree[0], nltk.Tree):
            label = re.split('-|=', tree.label())[0]
        root_arc_list = parent_arc + [label]
        root_arc = '+'.join(root_arc_list)
        if len(tree) == 1:
            root, arc, tag = tree2list(tree[0], parent_arc=root_arc_list)
        elif len(tree) == 2:
            c0, arc0, tag0 = tree2list(tree[0])
            c1, arc1, tag1 = tree2list(tree[1])
            root = [c0, c1]
            arc = arc0 + [root_arc] + arc1
            tag = tag0 + tag1
        else:
            c0, arc0, tag0 = tree2list(tree[0])
            c1, arc1, tag1 = tree2list(nltk.Tree('<empty>', tree[1:]))
            if bin == 0:
                root = [c0] + c1
            else:
                root = [c0, c1]
            arc = arc0 + [root_arc] + arc1
            tag = tag0 + tag1
        return root, arc, tag
    else:
        if len(parent_arc) == 1:
            parent_arc.insert(0, '<empty>')
        # parent_arc[-1] = '<POS>'
        del parent_arc[-1]
        return str(tree), [], ['+'.join(parent_arc)]


def get_distance(root):
    if isinstance(root, list):
        dist_list = []
        depth_list = []
        for child in root:
            dist, depth = get_distance(child)
            dist_list.append(dist)
            depth_list.append(depth)

        max_depth = max(depth_list)

        out = dist_list[0]
        for dist in dist_list[1:]:
            out.append(max_depth)
            out.extend(dist)
        return out, max_depth + 1
    else:
        return [], 1
