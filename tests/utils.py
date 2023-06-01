import torch
import tvm
from transformers import AutoTokenizer
from typing import Optional, List

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def get_tvm_model(const_params, vm):
    class Model:
        def __init__(self) -> None:
            self.tot_seq_len = 0
            self.kv_cache = vm["create_kv_cache"]()

        def forward(self, inputs: tvm.nd.array) -> tvm.nd.array:
            with torch.cuda.nvtx.range("tvm_model.forward"):
                self.tot_seq_len += inputs.shape[1]
                seq_len_shape = tvm.runtime.ShapeTuple([self.tot_seq_len])
                if inputs.shape[1] > 1:
                    logits, kv_cache = vm["prefill"](
                        inputs, seq_len_shape, self.kv_cache, const_params
                    )
                else:
                    logits, kv_cache = vm["decode"](
                        inputs, seq_len_shape, self.kv_cache, const_params
                    )
                self.kv_cache = kv_cache
                return logits

    model = Model()
    return model.forward


def get_pytorch_model(model, use_cache=True):
    def forward(inputs: torch.Tensor, past_key_values=None) -> (torch.Tensor, Optional[List[torch.FloatTensor]]):
        # NOTE: torch.inference_mode() is not supported with torch inductor yet.
        with torch.no_grad(): 
            out = model(inputs, use_cache=use_cache, past_key_values=past_key_values)
            return out.logits, out.past_key_values

    return forward


def get_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path)
