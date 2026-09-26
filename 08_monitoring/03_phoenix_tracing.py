# Monitoring 03 — Arize Phoenix: open-source tracing UI that runs on YOUR machine (no account, no key)
# Phoenix stores OpenTelemetry traces: every graph step, LLM call (prompt, answer, tokens, latency)
# and tool call, with errors highlighted in red.
#
# step 1 (terminal 1):  phoenix serve                  -> UI at http://localhost:6006
# step 2 (terminal 2):  python 08_monitoring/03_phoenix_tracing.py
# step 3: open http://localhost:6006 -> project "sa-lib-day2" -> click a trace
#
# we wire OpenTelemetry by hand (6 lines). the same lines send traces to ANY OpenTelemetry backend —
# change the endpoint to Langfuse, Jaeger, Grafana Tempo, ...
# other frameworks: pip install openinference-instrumentation-<crewai | openai-agents | google-adk | openai>
# and call its Instrumentor().instrument(tracer_provider=provider) the same way.

from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from agent_graph import QUESTIONS, build_agent

# --- tracing setup: where to send spans, and which project they belong to ---
provider = TracerProvider(resource=Resource({"openinference.project.name": "sa-lib-day2"}))
provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces")))
LangChainInstrumentor().instrument(tracer_provider=provider)   # every LangChain/LangGraph step becomes a span

# --- the agent, unchanged ---
agent = build_agent()
config = {"configurable": {"thread_id": "phoenix-demo"}}
for question in QUESTIONS:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config)
    print(result["messages"][-1].content)

provider.shutdown()   # flush spans before exit
print("\nopen http://localhost:6006 to see the traces (project: sa-lib-day2)")
