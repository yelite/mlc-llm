"""
A implementation of InferenceEngine that offload the inference loop to child process.
"""

import logging
import multiprocessing
import queue
from threading import Lock

from .base import (
    InferenceStepResult,
    Request,
    RequestId,
    RequestOutput,
    RequestState,
    ScopedInferenceEngine,
    SequenceOutput,
)
from .model_module import ConversationTemplate, Tokenizer, TokenizerModule
from .process_worker import (
    AddRequestsCommand,
    CancelRequestCommand,
    ShutdownCommand,
    run_generation_loop_worker,
)

logger = logging.getLogger(__name__)


class MultiProcessInferenceEngine(ScopedInferenceEngine):
    def __init__(
        self, tokenizer_module: TokenizerModule
    ):
        self.next_generation_output = None
        self.requests_lock = Lock()
        self.requests = dict[RequestId, RequestState]()

        self.tokenizer = tokenizer_module.tokenizer
        self.conversation_template = tokenizer_module.conversation_template

        self.mp_context = multiprocessing.get_context("spawn")
        self.command_queue = self.mp_context.Queue()
        self.result_queue = self.mp_context.Queue(maxsize=1)
        self.worker_process = self.mp_context.Process(
            target=run_generation_loop_worker,
            args=(self.command_queue, self.result_queue),
        )

    def start(self):
        self.worker_process.start()

    def stop(self):
        self.command_queue.put(ShutdownCommand())
        self.worker_process.join()

    def add(self, requests: list[Request]):
        if not self.worker_process.is_alive():
            raise RuntimeError("GenerationLoopWorker process is not running")

        new_request_states = []
        for req in requests:
            # TODO: verify that request id is unique
            if req.num_sequences > 1:
                raise RuntimeError("num_sequences > 1 is not supported for now")
            state = self._get_new_request_state(req)
            new_request_states.append(state)

        self.worker_process.put(AddRequestsCommand(request_states=new_request_states))

        with self.requests_lock:
            self.requests.update({s.request_id: s for s in new_request_states})

    def cancel(self, request_id: RequestId):
        if not self.worker_process.is_alive():
            raise RuntimeError("GenerationLoopWorker process is not running")
        self.worker_process.put(CancelRequestCommand(request_id))

    def wait_for_request(self, timeout_seconds=None) -> bool:
        if self.next_generation_output is not None:
            return True

        try:
            self.next_generation_output = self.result_queue.get(timeout=timeout_seconds)
            return True
        except queue.Empty:
            return False

    def step(self) -> InferenceStepResult:
        if self.next_generation_output is None:
            try:
                generation_output = self.result_queue.get_nowait()
            except queue.Empty:
                return InferenceStepResult([])
        else:
            generation_output = self.next_generation_output
            self.next_generation_output = None

        outputs = list[RequestOutput]()
        with self.requests_lock:
            for seq_output in generation_output.sequences:
                # TODO: support multi-sequence per request
                request_id = seq_output.id.request_id
                if request_id not in self.requests:
                    logger.warn(
                        "Unknown request %s from GenerationLoopWorkerOutput", request_id
                    )
                    continue

                state = self.requests[request_id]

                if seq_output.error is not None:
                    outputs.append(
                        RequestOutput(
                            request_id,
                            sequences=[],
                            error=seq_output.error,
                            num_prompt_tokens=state.prompt_len,
                        )
                    )
                    del self.requests[request_id]
                    continue

                state.next_start_position = len(state.token_ids)
                state.token_ids.extend(seq_output.new_tokens)

                delta = self._decode_last_output(state)
                state.output_text += delta

                outputs.append(
                    RequestOutput(
                        request_id,
                        sequences=[
                            SequenceOutput(
                                0,
                                delta=delta,
                                num_generated_tokens=(
                                    len(state.token_ids) - state.prompt_len
                                ),
                                finish_reason=seq_output.finish_reason,
                            ),
                        ],
                        num_prompt_tokens=state.prompt_len,
                    )
                )

        return None

    def _get_new_request_state(self, request: Request) -> RequestState:
        if request.debug_options.prompt is not None:
            prompt = request.debug_options.prompt
        else:
            prompt = self.conversation_template.apply(request.messages)

        prompt_tokens = self.tokenizer.encode(prompt)

        return RequestState(
            request_id=request.request_id,
            token_ids=prompt_tokens,
            prompt_len=len(prompt_tokens),
            next_start_position=0,
            sampling_params=request.sampling_params,
            stopping_criteria=request.stopping_criteria,
            debug_options=request.debug_options,
            output_text="",
        )

    def _decode_last_output(self, state: RequestState) -> str:
        if len(state.output_text):
            prefix_idx = max(0, state.next_start_position - 6)
        else:
            prefix_idx = state.next_start_position

        if prefix_idx == 0:
            return self.tokenizer.decode(state.token_ids)

        prefix = self.tokenizer.decode(
            state.token_ids[prefix_idx : state.next_start_position]
        )
        full = self.tokenizer.decode(state.token_ids[prefix_idx:])

        return full[len(prefix) :]
