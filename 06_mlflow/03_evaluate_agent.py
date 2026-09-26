# MLflow 03 — evaluating your agent
# "it worked once on my laptop" is not a test. mlflow.genai.evaluate runs your agent on a small
# dataset and scores every answer. here we use simple CODE scorers (free, no judge model needed).
# (MLflow also has LLM-as-a-judge scorers like Correctness() and Safety() — see the README.)
#
# run:   python 06_mlflow/03_evaluate_agent.py
# view:  mlflow ui --backend-store-uri sqlite:///mlflow.db  -> Experiments -> Evaluation runs

import mlflow
from mlflow.genai.scorers import scorer
from openai import OpenAI

import workshop_common as wc

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("SA-LIB Day2 agents")
mlflow.openai.autolog()

cfg = wc.get_llm_config()
client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)


# the agent under test (kept tiny on purpose — swap in any agent from the other folders)
def answer(question: str) -> str:
    resp = client.chat.completions.create(model=cfg.model, temperature=0, messages=[
        {"role": "system", "content": "You are a concise assistant. Answer in at most 40 words."},
        {"role": "user", "content": question}])
    return resp.choices[0].message.content


# the test set: inputs + what we expect
dataset = [
    {"inputs": {"question": "What is the capital of Libya?"}, "expectations": {"must_contain": "Tripoli"}},
    {"inputs": {"question": "What does RAG stand for in AI?"}, "expectations": {"must_contain": "retrieval"}},
    {"inputs": {"question": "Explain what an AI agent is in two sentences."}, "expectations": {"must_contain": "tool"}},
]


# scorers: plain python functions that grade one answer
@scorer
def contains_expected(outputs: str, expectations: dict) -> bool:
    return expectations["must_contain"].lower() in outputs.lower()


@scorer
def is_concise(outputs: str) -> bool:
    return wc.word_count(outputs) <= 40


results = mlflow.genai.evaluate(data=dataset, predict_fn=answer, scorers=[contains_expected, is_concise])
print("\naggregate metrics:")
for name, value in sorted(results.metrics.items()):
    print(f"  {name}: {value}")
