import torch
from transformers import AutoTokenizer, AutoModel

from config import bert_fine_tune_layers, device, bert_size
from model_utils import num_params
from string_embedder import StringEmbedder


class BertEmbedder(StringEmbedder):

    """
    sizes available:
                tiny (L=2, H=128)
                mini (L=4, H=256)
                small (L=4, H=512)
                medium (L=8, H=512)
    """

    def __init__(self):
        super().__init__()
        model_name = "prajjwal1/bert-" + bert_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        print("bert config:",  self.model.config.__dict__)
        self.dims = self.model.config.hidden_size
        self.set_trainable_params()
        print("Loaded bert model with", self.dims, "dims and ", num_params(self), "params")

    def set_trainable_params(self):
        def is_in_fine_tune_list(name):
            if name == "":  # full model is off by default
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
        # print("in ids:", input_ids.size())

        if input_ids.size(-1) > 512:
            # return self.get_windowed_attention(input_ids)
            raise TooManyTokens("too many tokens:", input_ids.size(-1))
        # attention_mask = encoding["attention_mask"].to(dev())
        # print("input ids:", input_ids.size())
        if self.fine_tune:
            out = self.model(input_ids=input_ids)#, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                out = self.model(input_ids=input_ids)#, attention_mask=attention_mask)
        last_hidden_state = out["last_hidden_state"]
        # if not self.fine_tune:
        #     last_hidden_state = last_hidden_state.detach()
        # print("last:", last_hidden_state.size())
        return last_hidden_state


class TooManyTokens(Exception):
    pass



if __name__ == "__main__":
    embedder = BertEmbedder()
    embedder.embed("hello world . I am Groot!")
    # embedder.embed(["hello world . I am Groot!", "yoyoyo"])