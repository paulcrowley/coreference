"""
A basic coreference resolver for the most common coreference types, using only nltk.
"""
import logging
from typing import List

import nltk

from .schemas import  Document, Mention
from coreference.matchers import (
    RelexiveResolver,
    RefDeterminerResolver,
    OneResolver,
    PronounResolver,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# TODO: determine if these could be added to the docker container.
nltk.download('averaged_perceptron_tagger')  # needed for RefDets
nltk.download('names')
nltk.download('maxent_ne_chunker')
nltk.download('treebank')
nltk.download('words')


matchers = [
    RelexiveResolver,
    PronounResolver,
    RefDeterminerResolver,
    OneResolver,
]


SEARCH_RANGE = 2


class CoreferenceResolver:
    """
    Loop through matching strategies to look for appropriate local antecedent for every
    anaphoric element. The matching strategies are prioritized in terms of precision.
    Bail after creating a single match initially. Afterwards pool matches.
    """

    @classmethod
    def resolve_coreferences(cls, raw_doc: str):
        """Return coreference pairs from input raw document text"""
        doc = Document(raw_doc)
        parsed_sents = doc.parsed_sents
        mentions = cls.create_doc_mentions(parsed_sents)

        logger.info(f"Mentions: {[mention.raw for mention in mentions]}")
        logger.info("\n------finding coref pairs-----\n")

        coref_pairs = []
        for mention in mentions:
            m_i = mentions.index(mention)
            candidates = mentions[:m_i]
            candidates = [
                candidate for candidate in candidates
                if mention.tree_i in range((mention.tree_i - SEARCH_RANGE), mention.tree_i)
            ]
            for candidate in candidates:
                if mention != candidate:
                    coref_pair = cls._match_antecedent(mention, candidate, parsed_sents)
                    if coref_pair:
                        coref_pairs.append(coref_pair)
                else:
                    continue

        return doc, coref_pairs


    @classmethod
    def _match_antecedent(cls, mention, candidate, parsed_sents):
        coref_pairs = []
        local = False
        if mention.tree_i == candidate.tree_i:
            local = True
        for matcher in matchers:
            match = matcher.find_antecedent(mention, candidate, parsed_sents, local)
            if match:
                coref_pairs.append((mention, candidate))
        return coref_pairs


    def pool_coreferences(self, coref_pairs):
        """Pool all coref pairs into bigger tuples based on common coreferences"""
        # NOTE: this will allow for exact matches, just need to make sure we're checking for
        # are all lowercase when pooling, and exclude pronouns"
        pass


    @classmethod
    def get_sent_nps(cls, tree):
        """Return all NP subtrees from a given tree structure"""
        listOfSubTrees = list(tree.subtrees())
        sentNPs = []
        for i in range(len(listOfSubTrees)):
            subtree1 = listOfSubTrees[i]
            if subtree1.label() in ['NP','PRP$']:
                sentNPs.append(subtree1)

        cls._filter_sent_nps(sentNPs)
        return sentNPs

    @staticmethod
    def _filter_sent_nps(sentNPs):
        """This function is needed to exclude embedded modified NPs in complex NPs from consideration,
         e.g. to exclude [the man] from [[the man] from France] from consideration in coreference."""

        filtered_NPs = []
        for i in range(len(sentNPs)):
            NP1 = sentNPs[i]
            if i == 0:
                filtered_NPs.append(NP1)
        for i in range(len(sentNPs)):
            NP1 = sentNPs[i]
            for j in range(len(sentNPs)):
                NP2 = sentNPs[j]
                if j == i - 1:
                    if NP2.leaves()[:len(NP1.leaves())] == NP1.leaves():
                        if len(NP2.leaves()) > NP1.leaves():
                            if nltk.pos_tag(NP2.leaves()[len(NP1.leaves()) + 1])[1] in ['IN', 'WDT']:
                            #condition covers preposition phrases and relative clauses
                                break
                    if NP2.leaves()[:len(NP1.leaves())] != NP1.leaves():
                        filtered_NPs.append(NP1)

        return filtered_NPs

    @classmethod
    def create_doc_mentions(cls, parsed_trees) -> List[Mention]:
        """Create Mention objects from each parsed sentence in doc"""
        doc_mentions = []
        t_i = 1
        for tree in parsed_trees:
            tree_nps = cls.get_sent_nps(tree)
            m_i = 1
            for np in tree_nps:
                mention = Mention(tree=np, i=m_i, tree_i=t_i)
                doc_mentions.append(mention)
                m_i += 1
            t_i += 1
        return doc_mentions

    @staticmethod
    def convert_leaf_encodings(mention):
        #convert from unicode format to ascii (no 'u' prefix) and creates string corresponding to each
        NPAscii = [leaf.encode("ascii") for leaf in mention.tree.leaves()]
        return ' '.join(NPAscii)



