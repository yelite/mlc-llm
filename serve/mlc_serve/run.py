import argparse
import os

import uvicorn

from mlc_llm import utils

from .api import create_app
from .engine import AsyncEngineConnector
from .engine.local import LocalProcessInferenceEngine
from .model.paged_cache_model import PagedCacheModelModule


def parse_args():
    # Example
    # python build.py --model vicuna-v1-7b --quantization q4f16_ft --use-cache=0 --max-seq-len 768 --batched
    # python tests/python/test_batched.py --local-id vicuna-v1-7b-q4f16_ft
    #
    # For Disco:
    # python build.py --model vicuna-v1-7b --quantization q0f16 --use-cache=0 --max-seq-len 768  --batched --build-model-only --num-shards 2
    # python build.py --model vicuna-v1-7b --quantization q0f16 --use-cache=0 --max-seq-len 768  --batched --convert-weight-only
    # /opt/bin/cuda-reserve.py  --num-gpus 2 python -m mlc_serve --local-id vicuna-v1-7b-q0f16 --num-shards 2
    #
    # Profile the gpu memory usage, and use the maximum number of cache blocks possible:
    # /opt/bin/cuda-reserve.py  --num-gpus 2 python -m mlc_serve --local-id vicuna-v1-7b-q0f16 --num-shards 2 --max-num-batched-tokens 2560 --max-input-len 256

    args = argparse.ArgumentParser()
    args.add_argument("--local-id", type=str, required=True)
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--num-shards", type=int, default=1)
    args.add_argument("--max-num-batched-tokens", type=int, default=-1)
    args.add_argument("--max-input-len", type=int, default=-1)
    parsed = args.parse_args()
    parsed.model, parsed.quantization = parsed.local_id.rsplit("-", 1)
    utils.argparse_postproc_common(parsed)
    parsed.artifact_path = os.path.join(
        parsed.artifact_path, f"{parsed.model}-{parsed.quantization.name}-batched"
    )
    return parsed


def run_server():
    args = parse_args()
    model_module = PagedCacheModelModule(
        args.model,
        args.artifact_path,
        args.quantization.name,
        args.num_shards,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_input_len=args.max_input_len,
    )

    engine = LocalProcessInferenceEngine(model_module)
    connector = AsyncEngineConnector(engine)
    app = create_app(connector)
    uvicorn.run(
        app,
        reload=False,
        access_log=False,
    )


if __name__ == "__main__":
    run_server()
