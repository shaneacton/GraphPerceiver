from typing import Tuple, List

from transformers import TokenSpan, BatchEncoding, PreTrainedTokenizerBase

from Code.Utils.spacy_utils import get_entity_char_spans
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

    safe_passage_words = [pw.replace("Ġ", "").replace("▁", "").lower() for pw in passage_words]
    safe_special_words = [[sw.replace("Ġ", "").replace("▁", "").lower() for sw in sws] for sws in special_words]
    special_letters = ["".join(sws) for sws in safe_special_words]

    token_spans = []

    for special_lets in special_letters:  # for each special ent
        """
            here special letters is a spaceless, lowercase textblob representing a candidate or query subject
            to match with a passage entity, all the chars must match up with the passage entities chars
            this is done because upper vs lower case can change how words are broken
            eg: (Olympic) vs (oly mp ic). As such, the token ids would not align
        """
        if len(special_lets) == 0:
            print("empty special ent. cands:", candidate_words, "query:", subject_words, "safe words:", safe_passage_words)
            print("query OG:", example.query)
            continue

        first_char = special_lets[0]
        match_indices = [i for i, pw in enumerate(safe_passage_words) if len(pw) > 0 and pw[0] == first_char]
        # all the passage words which begin with the first letter of our special ent charblob
        for i in match_indices:  # for each passage word starting with right letter
            j = i  # we will step through the following words
            pass_c = 0  # letter by letter
            # in order to see if we can find an exact match
            match_word = safe_passage_words[i]

            for spec_c in range(len(special_lets)):  # for each char in our special blob
                while j < len(safe_passage_words) and len(safe_passage_words[j]) == 0:  # ff empty passage words
                    j += 1
                    pass_c = 0  # start at beginning of new word
                if j == len(safe_passage_words):  # run out of words in passage
                    break

                match_word = safe_passage_words[j]  # load next passage word

                if special_lets[spec_c] != match_word[pass_c]:  # match failed on latest letter
                    break  # next match index

                pass_c += 1  # next letter in match word

                if spec_c == len(special_lets) -1:  # last letter, all matched up till here
                    if pass_c == len(match_word):  # just finished the pass word
                        # exact match!
                        # +1 for the CLS token at the start of each passage
                        token_spans.append(TokenSpan(i + 1, j + 2))
                        # print("found match! spec:", special_lets, " pass words:", safe_passage_words[i:j])
                    # else the spec word is over, but the match word is not, so no match!
                    break

                if pass_c == len(match_word):  # last letter, move onto next match word
                    j += 1
                    if j == len(safe_passage_words):  # run out of words in passage
                        break
                    pass_c = 0
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