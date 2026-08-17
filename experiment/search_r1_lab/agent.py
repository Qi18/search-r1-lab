from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from .protocol import extract_answer, extract_search_query, trim_to_first_action
from .retrieval import DenseRetriever, format_results


PROMPT = """Answer the question using the following protocol.
Reason inside <think> and </think>. If external knowledge is needed, emit
<search>your query</search>. Search results will be returned inside
<information> and </information>. Finish with a short
<answer>final answer</answer>.

Question: {question}
"""


class StopOnTags(StoppingCriteria):
    def __init__(self, tokenizer, tags: tuple[str, ...]) -> None:
        self.targets = [
            torch.tensor(tokenizer.encode(tag, add_special_tokens=False), dtype=torch.long)
            for tag in tags
        ]

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        for target in self.targets:
            if input_ids.shape[1] < target.numel():
                continue
            if torch.equal(input_ids[0, -target.numel() :].cpu(), target):
                return True
        return False


@dataclass
class AgentConfig:
    max_search_turns: int = 2
    max_new_tokens: int = 256
    topk: int = 3


class SearchR1Agent:
    def __init__(
        self,
        model_path: str,
        retriever: DenseRetriever,
        device: str,
        config: AgentConfig,
    ) -> None:
        self.retriever = retriever
        self.device = torch.device(device)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()
        self.stop = StoppingCriteriaList(
            [StopOnTags(self.tokenizer, ("</search>", "</answer>"))]
        )

    def _render_prompt(self, question: str) -> str:
        prompt = PROMPT.format(question=question.strip())
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        return prompt

    @torch.inference_mode()
    def answer(self, question: str, *, search_enabled: bool) -> dict:
        prompt = self._render_prompt(question)
        transcript = ""
        search_events: list[dict] = []
        start = time.perf_counter()
        input_token_count = 0
        output_token_count = 0

        for turn in range(self.config.max_search_turns + 1):
            encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_token_count += int(encoded["input_ids"].shape[1])
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                stopping_criteria=self.stop,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            generated_tokens = output[0, encoded["input_ids"].shape[1] :]
            output_token_count += int(generated_tokens.shape[0])
            generated = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True,
            )
            generated = trim_to_first_action(generated)
            transcript += generated
            prompt += generated

            if extract_answer(generated):
                break

            query = extract_search_query(generated)
            if query is None:
                break

            event = {
                "query": query,
                "retriever_requested": False,
                "results": [],
            }
            search_events.append(event)
            if not query or turn >= self.config.max_search_turns:
                break

            if search_enabled:
                event["results"] = self.retriever.search(query, self.config.topk)
                event["retriever_requested"] = True
            information = (
                format_results(event["results"])
                if search_enabled
                else "The search tool is disabled for this baseline."
            )
            observation = f"\n\n<information>{information}</information>\n\n"
            transcript += observation
            prompt += observation

        retriever_request_count = sum(
            event["retriever_requested"] for event in search_events
        )
        return {
            "prediction": extract_answer(transcript),
            "trajectory": transcript,
            "search_events": search_events,
            "generated_search_count": len(search_events),
            "retriever_request_count": retriever_request_count,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "latency_seconds": time.perf_counter() - start,
        }
