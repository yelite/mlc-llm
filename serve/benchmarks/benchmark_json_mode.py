import argparse
import asyncio
import time
import aiohttp

from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from random import random, choice


class Answer(BaseModel):
    answer: str


class User(BaseModel):
    name: str
    age: int


class Snow(BaseModel):
    color: str


class SnowList(BaseModel):
    snow: List[Snow]


class Car(BaseModel):
    make: str
    color: Optional[str]


class DayPlan(BaseModel):
    region: str
    activities: List[str]
    expense: float


class TravelPlan(BaseModel):
    plan: List[DayPlan]


class Person(BaseModel):
    name: str
    age: int


TESTS = [
    ("Hello?", Answer),
    ("His name is Jason and he is 25 years old", User),
    ("We observed snow colors of white and yellow", SnowList),
    ("The make of the car is toyota, no color specified", Car),
    ("The make of the car is subaru and its color is black", Car),
    ("One week travel plan to Switzerland", TravelPlan),
    (
        "Alice is a 30-year-old individual who brings a wealth of "
        "experience and knowledge to any situation. With her years "
        "of expertise and youthful energy, she is a valuable asset "
        "in any team or project. Her dedication and passion shine "
        "through in everything she does, making her a reliable and "
        "trustworthy colleague. Alice's commitment to excellence "
        "and continuous growth sets her apart, ensuring success "
        "in all her endeavors.",
        Person,
    ),
]


def create_request(
    content: str, schema: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    data = {
        "model": "placeholder",
        "temperature": 0,
        "max_tokens": 512,
        "messages": [{"role": "user", "content": content}],
    }

    if schema:
        data["response_format"] = {"type": "json_object", "schema": schema}

    return data


async def fetch(url, data):
    headers = {"Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers) as response:
            return await response.json()


async def run(endpoint: str, ratio: float, num_requests: int) -> float:
    url = f"{endpoint}/v1/chat/completions"
    tasks = []
    models = []

    for i in range(num_requests):
        content, model = choice(TESTS)
        args = {"content": content}
        if ratio > random():
            args["schema"] = model.model_json_schema()
        m = model if "schema" in args else None
        models.append(m)
        data = create_request(**args)
        tasks.append(fetch(url=url, data=data))

    start = time.perf_counter()

    results = await asyncio.gather(*tasks)

    end = time.perf_counter()

    elapsed_time = end - start

    # Count tokens
    tokens = 0
    fails = 0
    total = 0
    for r, m in zip(results, models):
        tokens += r["usage"]["total_tokens"]
        if m:
            total += 1
            try:
                jsonstr = r["choices"][0]["message"]["content"]
                m.model_validate_json(jsonstr, strict=True)
            except:
                fails += 1

    return tokens, elapsed_time, total, fails


async def main(args: argparse.Namespace):
    total_tokens, elapsed_time, total, fails = await run(
        args.endpoint, args.ratio, args.num_requests
    )

    req_per_sec = args.num_requests / elapsed_time
    print(
        f"Engine Throughput: {req_per_sec:.2f} requests/s, "
        f"{total_tokens / elapsed_time:.2f} tokens/s"
    )
    print(f"JSON mode (failed/total) requests: {fails}/{total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:10001",
        help="The endpoint url",
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="The ratio of JSON mode traffic, 0 begin none and 1 being 100%",
    )

    parser.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="The number of requests",
    )

    args = parser.parse_args()

    asyncio.run(main(args))
