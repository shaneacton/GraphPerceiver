from typing import Tuple, List

from transformers import TokenSpan, BatchEncoding

from spacy_utils import get_entity_char_spans


class Wikipoint:

    def __init__(self, example, tokeniser=None):
        supports = example["supports"]
        self.supports = supports
        self.answer = example["answer"]
        self.candidates = example["candidates"]
        self.query = example["query"]

        supp_encs = [tokeniser(supp) for supp in supports]
        self.ent_token_spans: List[List[Tuple[int]]] = get_transformer_entity_token_spans(supp_encs, supports)

    @property
    def relation(self):
        return self.query.split(" ")[0]

    @property
    def query_subject(self):
        return " ".join(self.query.split(" ")[1:])  # the rest

    def __repr__(self):
        ex = "Wiki example:\n"
        ex += "Query:" + self.query + "\n"
        ex += "Candidates:" + ", ".join(self.candidates) + "\n\n"
        ex += "Answer:" + self.answer + "\n"
        ex += "Supports:\n" + "\n\t".join(self.supports)
        return ex


def get_transformer_entity_token_spans(support_encodings, supports) -> List[List[Tuple[int]]]:
    """
        token_spans is indexed list[support_no][ent_no]
        summaries is a flat list
    """
    token_spans: List[List[Tuple[int]]] = []

    for s, support in enumerate(supports):
        """get entity node embeddings"""
        ent_c_spans = get_entity_char_spans(support)
        support_encoding = support_encodings[s]

        ent_token_spans: List[Tuple[int]] = []
        for e, c_span in enumerate(ent_c_spans):
            """clips out the entities token embeddings, and summarises them"""
            try:
                ent_token_span = charspan_to_tokenspan(support_encoding, c_span)
            except Exception as ex:
                print("cannot get ent ", e, "token span. in supp", s)
                print(ex)
                continue
            ent_token_spans.append(ent_token_span)

        token_spans.append(ent_token_spans)

    return token_spans


def charspan_to_tokenspan(encoding: BatchEncoding, char_span: Tuple[int]) -> TokenSpan:
    start = encoding.char_to_token(char_index=char_span[0], batch_or_char_index=0)
    if start is None:
        raise Exception("cannot get token span from charspan:", char_span, "given:", encoding.tokens())

    recoveries = [-1, 0, -2, -3]  # which chars to try. To handle edge cases such as ending on dbl space ~ '  '
    end = None
    while end is None:
        if len(recoveries) == 0:
            raise Exception(
                "could not get end token span from char span:" + repr(char_span) + " num tokens: " + repr(
                    len(encoding.tokens())) + " ~ " + repr(encoding))

        offset = recoveries.pop(0)
        end = encoding.char_to_token(char_index=char_span[1] + offset, batch_or_char_index=0)

    span = TokenSpan(start - 1, end)  # -1 to discount the <s> token
    return span