from nltk.parse.stanford import StanfordParser


# setting up the standford parser (set up for my mac)
# stanford parser downloaded from https://nlp.stanford.edu/software/lex-parser.shtml


PARSE_JAR = "/Users/pcrowley/Downloads/stanford-parser-4.0.0/stanford-parser.jar"
PARSE_MODEL = "/Users/pcrowley/Downloads/stanford-parser-4.0.0/stanford-parser-4.0.0-models.jar"


class ParserManager:
    def __init__(self, parse_obj=StanfordParser, jar=None, model=None):
        self.jar = jar
        self.model = model
        self.parser = parse_obj(path_to_jar=jar, path_to_models_jar=model)

    def parse_sentence(self, sents):
        """Tokenize sentences from a document and parse each sentence"""
        raw_parse_outputs = [self.parser.raw_parse(sent) for sent in sents]
        parse_list = [list(itr_obj)[0] for itr_obj in raw_parse_outputs]
        return parse_list

    def draw_doc_trees(self, parsed_docs):
        """Draw tree structure"""
        for tree in parsed_docs:
            tree.draw()


parser_mgr = ParserManager(StanfordParser, PARSE_JAR, PARSE_MODEL)
