import nltk

from .parser import parser_mgr


class Document:
    def __init__(self, raw_text):
        self.raw_text = raw_text
        self.sents = nltk.sent_tokenize(self.raw_text)
        self.parsed_sents = [parser_mgr.parse_sentence(sent) for sent in self.sents]


class Mention:
    def __init__(self, raw=None, tree=None, i=None, tree_i=None):
        self.raw = raw
        self.tree = tree
        self.nouns = self.get_nouns()
        self.i = i
        self.tree_i = tree_i

    def get_nouns(self):
        # returns all nouns in a tree. Used for head noun matching in RDT component (E)
        nounLevelList = []
        for subtree in self.tree.subtrees():
            if subtree.label() in ['NN', 'NNS']:
                nounLevelList.append(subtree.leaves()[0])
        return nounLevelList