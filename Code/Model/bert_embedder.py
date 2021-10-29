import torch
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizerBase, BigBirdModel, RobertaModel, \
    RobertaTokenizer, RobertaTokenizerFast, BigBirdTokenizerFast

from Config.options import bert_fine_tune_layers, device, bert_size, embedder_name
from Code.Utils.model_utils import num_params
from Code.string_embedder import StringEmbedder


class BertEmbedder(StringEmbedder):

    """
    Model: prajjwal1/bert-
    sizes available:
                tiny (L=2, H=128)
                mini (L=4, H=256)
                small (L=4, H=512)
                medium (L=8, H=512)

    Model: google/bigbird-roberta-base
    Model: roberta-base
    """

    def __init__(self):
        super().__init__()
        model_name = embedder_name
        if "prajjwal1" in model_name:
            model_name += bert_size
            self.model = AutoModel.from_pretrained(model_name)
            self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)

        elif "bigbird" in model_name:
            self.model = BigBirdModel.from_pretrained(model_name, block_size=16, num_random_blocks=2)
            self.tokenizer: BigBirdTokenizerFast = BigBirdTokenizerFast.from_pretrained(model_name)

        elif "roberta" in model_name:
            self.model = RobertaModel.from_pretrained(model_name)
            self.tokenizer: RobertaTokenizerFast = RobertaTokenizerFast.from_pretrained(model_name)

        print("bert config:",  self.model.config.__dict__)
        self.dims = self.model.config.hidden_size
        self.set_trainable_params()
        print("Loaded bert model with", self.dims, "dims and ", num_params(self), "params")

    def set_trainable_params(self):
        def is_in_fine_tune_list(name):
            if name == "" and "" not in bert_fine_tune_layers:  # full model is off by default
                return False

            for l in bert_fine_tune_layers:
                if l in name:
                    return True
            return False

        for param in self.model.parameters():
            """all params are turned off. then we selectively reactivate grads"""
            param.requires_grad = False
        for n, m in self.model.named_modules():
            if not is_in_fine_tune_list(n):
                continue
            for param in m.parameters():
                param.requires_grad = True

    def embed(self, string):
        encoding = self.tokenizer(string, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)

        if input_ids.size(-1) > 512:
            raise TooManyTokens("too many tokens:", input_ids.size(-1))
        # attention_mask = encoding["attention_mask"].to(dev())
        out = self.model(input_ids=input_ids)#, attention_mask=attention_mask)

        last_hidden_state = out["last_hidden_state"]
        return last_hidden_state, encoding


class TooManyTokens(Exception):
    pass


if __name__ == "__main__":
    embedder = BertEmbedder()
    embedder.embed("hello world . I am Groot!")
    # embedder.embed(["hello world . I am Groot!", "yoyoyo"])