"""
A implementation of InferenceEngine that offload the inference loop to child process.
"""

import logging
import multiprocessing
import queue
from collections import deque
from threading import Condition, Lock

from .base import (
    DebugOptions,
    FinishReason,
    InferenceStepResult,
    Request,
    RequestId,
    RequestOutput,
    RequestState,
    SamplingParams,
    ScopedInferenceEngine,
    SequenceOutput,
    StoppingCriteria,
)
from .model_module import (
    ConversationTemplate,
    DecodeRequest,
    ModelModule,
    PrefillRequest,
    SequenceId,
    Tokenizer,
)
from .process_worker import (
    AddRequestsCommand,
    CancelRequestCommand,
    ShutdownCommand,
    run_generation_loop_worker,
)

logger = logging.getLogger(__name__)


class MultiProcessInferenceEngine(ScopedInferenceEngine):
    def __init__(
        self, tokenizer: Tokenizer, conversation_template: ConversationTemplate
    ):
        self.next_generation_output = None

        self.tokenizer = tokenizer
        self.conversation_template = conversation_template

        self.mp_context = multiprocessing.get_context("spawn")
        self.command_queue = self.mp_context.Queue()
        self.result_queue = self.mp_context.Queue(maxsize=2)
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

        # TODO: convert generation output
        outputs = []
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
