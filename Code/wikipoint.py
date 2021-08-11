from typing import Tuple, List

from transformers import TokenSpan, BatchEncoding, PreTrainedTokenizerBase

from Code.Utils import get_entity_char_spans
from Config.options import use_detected_ents, use_special_ents


class Wikipoint:

    def __init__(self, example, tokeniser:PreTrainedTokenizerBase=None):
        supports = example["supports"]
        self.supports = supports
        self.answer = example["answer"]
        self.candidates = example["candidates"]
        self.query = example["query"]

        supp_encs = [tokeniser(supp) for supp in supports]
        if use_detected_ents:
            self.ent_token_spans: List[List[Tuple[int]]] = get_detected_entity_token_spans(supp_encs, supports)
        elif use_special_ents:
            self.ent_token_spans: List[List[Tuple[int]]] = get_special_entity_token_spans(self, supp_encs, tokeniser)
        else:
            raise Exception()

        # print("found ents:", [[supp_encs[d].tokens()[span[0]:span[1]] for span in doc] for d, doc in enumerate(self.ent_token_spans)])

    @property
    def relation(self):
        return self.query.split(" ")[0]

    @property
    def query_subject(self):
        return " ".join(self.query.split(" ")[1:])  # the rest

    def num_ents(self):
        return sum([len(e) for e in self.ent_token_spans])

    def get_flat_ent_id(self, doc_id, ent_in_doc_id):
        count = 0
        for d in range(len(self.ent_token_spans)):
            for ed in range(len(self.ent_token_spans[d])):
                if ed >= ent_in_doc_id and d >= doc_id:
                    return count
                count += 1
        raise Exception()

    def __repr__(self):
        ex = "Wiki example:\n"
        ex += "Query:" + self.query + "\n"
        ex += "Candidates:" + ", ".join(self.candidates) + "\n\n"
        ex += "Answer:" + self.answer + "\n"
        ex += "Supports:\n" + "\n\t".join(self.supports)

        return ex


def get_special_entity_token_spans(example, support_encodings, tokeniser:PreTrainedTokenizerBase) -> List[List[Tuple[int]]]:
    """returns a 2d list indexed [supp_id][ent_in_supp]"""
    all_token_spans: List[List[Tuple[int]]] = []
    for s, _ in enumerate(example.supports):
        doc_token_spans = get_special_entity_token_spans_from_doc(example, support_encodings[s], tokeniser)
        all_token_spans.append(doc_token_spans)
    return all_token_spans


def get_special_entity_token_spans_from_doc(example, support_encoding: BatchEncoding,
                                            tokeniser: PreTrainedTokenizerBase) -> List[Tuple[int]]:
    passage_words: List[str] = support_encoding.tokens()[1:-1]
    subject_words: List[str] = tokeniser(example.query_subject).tokens()[1:-1]
    candidate_words: List[List[str]] = [tokeniser(cand).tokens()[1:-1] for cand in example.candidates]
    special_words: List[List[str]] = candidate_words + [subject_words]

    token_spans = []
    for specials in special_words:
        if len(specials) <= 0:
            continue
        start = specials[0]
        indices = [i for i, x in enumerate(passage_words) if x == start]
        for i in indices:
            corr_passage_words = passage_words[i:i+len(specials)]
            if corr_passage_words == specials:
                # +1 for the CLS token at the start of each passage
                token_spans.append(TokenSpan(i + 1, i+len(specials) + 1))

    return token_spans


def get_detected_entity_token_spans(support_encodings, supports) -> List[List[Tuple[int]]]:
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