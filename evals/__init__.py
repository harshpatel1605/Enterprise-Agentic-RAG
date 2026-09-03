import sys
try:
    import langchain_google_vertexai.chat_models
    sys.modules.setdefault("langchain_community.chat_models.vertexai", langchain_google_vertexai.chat_models)
except ImportError:
    pass

from evals.pipeline import run_pipeline, load_golden_dataset
from evals.guardrails_eval import run_guardrails_eval, compute_guardrails_metrics
from evals.metrics import run_all_metrics
