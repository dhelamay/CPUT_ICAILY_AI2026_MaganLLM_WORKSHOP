# MLflow 04 — package the agent as a versioned MLflow model, load it back, and serve it
# logging = saving the code + dependencies + an example, with a version you can roll back to.
#
# run:    python 06_mlflow/04_package_agent.py
# serve:  mlflow models serve -m "models:/<model_id>" -p 5001 --env-manager local
#         (the exact command is printed at the end)
# call:   curl -X POST localhost:5001/invocations -H "Content-Type: application/json" \
#              -d '{"input": [{"role": "user", "content": "what is an AI agent?"}]}'

from pathlib import Path

import mlflow

HERE = Path(__file__).resolve().parent
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("SA-LIB Day2 agents")

with mlflow.start_run(run_name="package-agent"):
    info = mlflow.pyfunc.log_model(
        name="workshop_agent",
        python_model=str(HERE / "agent_model.py"),            # "models from code": log the FILE
        code_paths=[str(HERE / "workshop_common.py")],         # helper module it imports
        input_example={"input": [{"role": "user", "content": "what is an AI agent?"}]},
        pip_requirements=["mlflow", "openai", "python-dotenv", "ddgs"],
    )
print("logged:", info.model_uri)

# load it back exactly like a deployed service would, and call it
agent = mlflow.pyfunc.load_model(info.model_uri)
response = agent.predict({"input": [{"role": "user", "content": "search for the latest langchain version"}]})
print(response["output"][0]["content"][0]["text"])

print(f'\nserve it:  mlflow models serve -m "{info.model_uri}" -p 5001 --env-manager local')
