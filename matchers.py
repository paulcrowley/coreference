import logging

import nltk
from nltk.corpus import wordnet as wn       # for gender pronoun matching
from nltk import ne_chunk     # for NER tagger
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from coreference import words

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResolverAbstract:
    @classmethod
    def find_antecedent(self, mention, candidate, parsed_sents, local):
        return None

    @staticmethod
    def get_nouns(tree):
        #returns all nouns in a tree. Used for head noun matching in RDT component (E)
        nounLevelList = []
        for subtree in tree.subtrees():
            if subtree.label() in ['NN', 'NNS']:
                nounLevelList.append(subtree.leaves()[0])
        return nounLevelList


class OneResolver(ResolverAbstract):
    @classmethod
    def find_antecedent(cls, mention, candidate, parsed_sents, local):
        """Match 'the one': 'A big dog and a small dog came in. The big one was friendly'"""
        if len(candidate.tree.leaves()) >= 3:
            if candidate.tree.leaves()[0].lower() == "the":
                if any(
                    leaf in ["one", "ones"] for leaf in candidate.tree.leaves()
                ):
                    reduced_cand_leafs = [
                        leaf for leaf in candidate.tree.leaves() if leaf not in ["the", "one", "ones"]
                    ]
                    if all(
                        leaf in mention.leaves() for leaf in reduced_cand_leafs
                    ):
                        logger.info(f"'THE_ONE'_MATCH: {(mention.raw, candidate.raw)}")
                        return True
        return False


class RefDeterminerResolver(ResolverAbstract):
    @classmethod
    def find_antecedent(cls, mention, candidate, parsed_sents, local):
        """Example: 'A woman from France was here. The woman was tall'"""
        mention_nouns = cls.get_nouns(mention)
        candidate_nouns = cls.get_nouns(candidate)
        if len(mention_nouns) > 0:
            if all(
                leaf in candidate.tree.leaves()
                for leaf in mention.tree.leaves()[1:]
            ):
                if mention_nouns[0] == candidate_nouns[0]:
                    logger.info(f"NOUN_PHRASE_MATCH: {(mention.raw, candidate.raw)}")
                    return True

        return False


class RelexiveResolver(ResolverAbstract):
    @classmethod
    def find_antecedent(cls, mention, candidate, parsed_sents, local):
        if mention.leaves()[0] in words.reflexives:
            if local:
                containing_tree = parsed_sents[mention.tree_i]
                for node in containing_tree.subtrees():
                    if node.label() == "S":
                        if candidate.i == 0:
                            if mention.tree in node.subtrees() and candidate.tree in node.subtrees():
                                if mention.tree_i == candidate.tree_i:
                                    logger.info(f"REFLEXIVE_PRONOUN_MATCH: {(mention.raw, candidate.raw)}")
                                    return True

        return False


class PronounResolver(ResolverAbstract):
    @classmethod
    def find_antecedent(cls, mention, candidate, parsed_sents, local):
        # Check for clausemate relation--bail if found
        if local:
            containing_tree = parsed_sents[mention.tree_i]
            for node in containing_tree.subtrees():
                if node.label() == "S" and not any(sub_node == "S" for sub_node in node.subtrees()):
                    if mention.tree in node.subtrees() and candidate.tree in node.subtrees():
                        return False

        male_pronoun_match = cls.resolve_male_pronouns(mention, candidate)
        female_pronoun_match = cls.resolve_female_pronouns(mention, candidate)
        plural_pronoun_match = cls.resolve_plural_pronoun(mention, candidate)
        it_match = cls.resolve_it(mention, candidate)

        match_attempts = [male_pronoun_match, female_pronoun_match, plural_pronoun_match, it_match]

        if any(match for match in match_attempts):
            return True

        return False


    @classmethod
    def resolve_male_pronouns(cls, mention, candidate):
        if candidate.leaves()[0].lower() in words.male_pronouns:
            return True
        elif all(pair[1] == "NNP" for pair in nltk.pos_tag(candidate.tree.leaves())):
            # Found a male-named antecedent
            if candidate.leaves()[0] in words.male_names:
                logger.info(f"MALE_PRONOUN_MATCH: {(mention.raw, candidate.raw)}")
                return True
        elif len(candidate.nouns) > 0:
            for node in candidate.tree.subtrees():
                if node.label() == 'NN' and node.leaves()[0] in words.total_male_nouns:
                    return True

        return False


    @classmethod
    def resolve_female_pronouns(cls, mention, candidate):
        if candidate.leaves()[0].lower() in words.female_pronouns:
            return True
        elif all(pair[1] == "NNP" for pair in nltk.pos_tag(candidate.tree.leaves())):
            # Found a female-named antecedent
            if candidate.leaves()[0] in words.female_names:
                logger.info(f"FEMALE_PRONOUN_MATCH: {(mention.raw, candidate.raw)}")
                return True
        elif len(candidate.nouns) > 0:
            for node in candidate.tree.subtrees():
                if node.label() == 'NN' and node.leaves()[0] in words.total_female_nouns:
                    return True
        return False


    @classmethod
    def resolve_plural_pronoun(cls, mention, candidate):
        if mention.tree.leaves()[0] in words.plural_pronouns:
            if candidate.tree.leaves()[0] in words.plural_pronouns:
                return True

            elif len(candidate.nouns) > 0:
                head_noun = candidate.nouns[0]
                if (pos_pair[1] == "NNS" for pos_pair in nltk.pos_tag(head_noun)):
                    logger.info(f"PLURAL_PRONOUN_MATCH: {(mention.raw, candidate.raw)}")
                    return True

        return False


    @classmethod
    def resolve_it(cls, mention, candidate):
        if mention.leaves()[0].lower() == "it":

            if candidate.leaves()[0] in ["It", "it"]:
                if not candidate.leaves()[0] in words.non_it_pronouns:
                    return True

            elif str(ne_chunk(pos_tag(word_tokenize(' '.join(candidate.leaves()))))[0])[0:3] != "(u'":
                # Check first if output of NE chunking is a non-NE mention--they all start with "(u"
                if str(ne_chunk(pos_tag(word_tokenize(' '.join(candidate.leaves()))))[0])[1:7] != "PERSON":
                    # Disqualify person names
                    if not candidate.leaves()[0] in words.person_names:
                        logger.info(f"IT_PRONOUN_MATCH: {(mention.raw, candidate.raw)}")
                        return True

            elif len(candidate.nouns) > 0:
                # Determine that that the candidate has a head noun (is not an NE)
                head_noun = candidate.nouns[0]
                head_noun_entry = wn.synsets(head_noun)[0]
                hypernyms = head_noun_entry.hypernym_paths()
                hypernyms = [x for l in hypernyms for x in l]
                person = wn.synset('person.n.01')
                if person not in hypernyms:
                    logger.info(f"IT_PRONOUN_MATCH: {(mention.raw, candidate.raw)}")
                    return True

        return False


