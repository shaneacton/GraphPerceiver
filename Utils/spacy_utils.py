from typing import List, Tuple, Dict

try:
    from spacy.tokens.span import Span

    try:
        import en_core_web_sm
        nlp = en_core_web_sm.load()
    except:
        print("failed to load en_core_web_sm, trying in cluster location")
        import spacy

        spacy.util.set_data_path('/home/sacton/.conda/envs/gnn_env/lib/python3.8/site-packages')
        nlp = spacy.load('en_core_web_sm')

except:
    print("spacy not installed. tokens only")


def _init_doc(doc, text):
    if not doc:
        doc = nlp(text)
    return doc


def get_char_span_from_spacy_span(span, doc) -> Tuple[int]:
    start_char = doc[span.start].idx
    end_char = doc[span.end -1].idx + len(doc[span.end -1])
    return start_char, end_char


def get_sentence_char_spans(text, doc=None) -> List[Tuple[int]]:
    doc = _init_doc(doc, text)
    sents = doc.sents
    sent_spans = [get_char_span_from_spacy_span(s, doc) for s in sents]

    return sent_spans, doc


def get_entity_char_spans(text, doc=None) -> List[Tuple[int]]:
    doc = _init_doc(doc, text)

    ent_spans = []
    for ent in doc.ents:
        ent_span = get_char_span_from_spacy_span(ent, doc)
        ent_spans.append(ent_span)
    return ent_spans


def get_noun_char_spans(text, doc=None):
    doc = _init_doc(doc, text)
    nouns = [n for n in doc.noun_chunks]
    return [get_char_span_from_spacy_span(n, doc) for n in nouns]


if __name__ == "__main__":
    # text = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very " \
    #            "close to the Manhattan Bridge which is visible from the window."
    # text = context(test_example)
    text = 'A Christian (or ) is a person who follows or adheres to Christianity, an Abrahamic, monotheistic religion based on the life and teachings of Jesus Christ. "Christian" derives from the Koine Greek word "Christós" (), a translation of the Biblical Hebrew term "mashiach".'
    print(text)
    doc = None
    # char_spans, doc = get_flat_entity_and_corefs_chars(text, doc=doc)
    # char_spans, doc = get_sentence_char_spans(text, doc=doc)
    # char_spans, doc = get_noun_char_spans(text, doc=doc)
    char_spans = get_noun_char_spans(text)
    for s in char_spans:
        print(text[s[0]: s[1]])