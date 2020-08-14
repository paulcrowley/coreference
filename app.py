from coreference.coref import CoreferenceResolver




if __name__ ==  "__main__":
    text = "John is here. He loves Mary. Mary loves herself."
    CoreferenceResolver.resolve_coreferences(text)