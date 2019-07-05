import zss

from dss_vae.structs import PhraseTree

try:
    from editdist import distance as strdist
except ImportError:
    def strdist(a, b):
        if a == b:
            return 0
        else:
            return 1


def weird_dist(A, B):
    return 10 * strdist(A, B)


class WeirdNode(object):

    def __init__(self, label):
        self.my_label = label
        self.my_children = list()

    @staticmethod
    def get_children(node):
        return node.my_children

    @staticmethod
    def get_label(node):
        return node.my_label

    def addkid(self, node, before=False):
        if before:
            self.my_children.insert(0, node)
        else:
            self.my_children.append(node)
        return self


def tree_to_weird(phrase_tree: PhraseTree):
    parent = WeirdNode(phrase_tree.symbol)
    if phrase_tree.leaf is not None:
        return parent
    for child in phrase_tree.children:
        child_node = tree_to_weird(child)
        parent.addkid(child_node)
    return parent


def file_to_weird(phrase_tree_file):
    tree_list = PhraseTree.load_treefile(phrase_tree_file)
    return [tree_to_weird(tree) for tree in tree_list]


def evaluate_tree_edit_distance(inputs_1, inputs_2, **kwargs):
    distance_list = []
    for i1, i2 in zip(inputs_1, inputs_2):
        distance_list.append(
            zss.simple_distance(
                i1, i2, WeirdNode.get_children, WeirdNode.get_label, weird_dist)
        )
    return distance_list


def evaluate_all(ori, tgt, pred):
    ori_list = file_to_weird(ori)
    tgt_list = file_to_weird(tgt)
    pred_list = file_to_weird(pred)
    ori_pred_dis = evaluate_tree_edit_distance(ori_list, pred_list)
    tgt_pred_dis = evaluate_tree_edit_distance(tgt_list, pred_list)
    print("Sim To Ori", sum(ori_pred_dis) / len(ori_pred_dis))
    print("Sim To Tgt", sum(tgt_pred_dis) / len(tgt_pred_dis))

# A = (
#     WeirdNode("f")
#         .addkid(WeirdNode("d")
#                 .addkid(WeirdNode("a"))
#                 .addkid(WeirdNode("c")
#                         .addkid(WeirdNode("b"))
#                         )
#                 )
#         .addkid(WeirdNode("e"))
# )
# B = (
#     WeirdNode("f")
#         .addkid(WeirdNode("d")
#                 .addkid(WeirdNode("a"))
#                 .addkid(WeirdNode("c")
#                         .addkid(WeirdNode("b"))
#                         .addkid(WeirdNode("c"))
#                         )
#                 )
#         .addkid(WeirdNode("e"))
# )
#
# dist = zss.simple_distance(
#     A, B, WeirdNode.get_children, WeirdNode.get_label, weird_dist)
#
# print(dist)
