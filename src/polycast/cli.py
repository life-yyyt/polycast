"""Command line entry point: `polycast forecast | serve | eval | simulate`."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone

from .audit import ForecastStore
from .corpus import build_retriever
from .evaluation import load_resolved, recalibrate, score_resolved
from .pipeline import ForecastPipeline
from .providers import MockProvider, OpenAICompatibleProvider
from .schemas import Horizon, Question
from .simulation import run_simulation


def _pipeline(args) -> ForecastPipeline:
    if os.getenv("POLYCAST_MODEL") or getattr(args, "model", None):
        provider = OpenAICompatibleProvider(
            model=os.environ.get("POLYCAST_MODEL") or args.model,
            api_key=os.getenv("POLYCAST_API_KEY"),
            base_url=os.getenv("POLYCAST_BASE_URL", "https://api.openai.com/v1"),
        )
    else:
        provider = MockProvider()
    return ForecastPipeline(
        provider=provider,
        retriever=build_retriever(args.corpus),
        store=ForecastStore(args.db),
        audit_dir=args.audit_dir,
    )


def cmd_forecast(args) -> int:
    resolve_by = (
        datetime.fromisoformat(args.resolve_by)
        if args.resolve_by
        else datetime.now(timezone.utc).replace(month=12, day=31)
    )
    question = Question(
        text=args.text,
        resolution_criteria=args.criteria,
        resolve_by=resolve_by,
        horizon=Horizon(args.horizon),
        reference_class=args.reference_class,
    )
    forecast = asyncio.run(_pipeline(args).run(question))
    if args.json:
        print(json.dumps(forecast.model_dump(mode="json"), indent=2, ensure_ascii=False))
    else:
        print(f"probability : {forecast.probability:.3f}")
        print(f"interval    : [{forecast.lower:.3f}, {forecast.upper:.3f}]")
        print(f"disagreement: {forecast.disagreement:.3f}")
        print(f"evidence    : {', '.join(e.doc_id for e in forecast.evidence) or 'none'}")
        for view in forecast.views:
            print(f"  {view.persona:<10} {view.probability:.3f}  {view.reasoning[:70]}")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("polycast.server:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_eval(args) -> int:
    resolved = load_resolved(args.dataset)
    pipeline = _pipeline(args)
    pairs = asyncio.run(score_resolved(pipeline, resolved))
    result = recalibrate(pairs)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_simulate(args) -> int:
    result = run_simulation(
        n_agents=args.agents,
        truth=args.truth,
        rounds=args.rounds,
        social_weight=args.social_weight,
        topology=args.topology,
        signal_noise=args.noise,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "truth": result.truth,
                "final_mean": result.final_mean,
                "final_spread": result.final_spread,
                "converged_round": result.converged_round,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="polycast", description=__doc__)
    parser.add_argument("--corpus", default=None, help="path to corpus json")
    parser.add_argument("--db", default="polycast.db", help="sqlite path")
    parser.add_argument("--audit-dir", default="runs", help="audit trace directory")
    parser.add_argument("--model", default=None, help="model id for a hosted provider")
    sub = parser.add_subparsers(dest="command", required=True)

    p_forecast = sub.add_parser("forecast", help="run one forecast")
    p_forecast.add_argument("--text", required=True)
    p_forecast.add_argument("--criteria", required=True)
    p_forecast.add_argument("--resolve-by", default=None)
    p_forecast.add_argument("--horizon", default="months", choices=[h.value for h in Horizon])
    p_forecast.add_argument("--reference-class", default=None)
    p_forecast.add_argument("--json", action="store_true")
    p_forecast.set_defaults(func=cmd_forecast)

    p_serve = sub.add_parser("serve", help="run the HTTP API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_eval = sub.add_parser("eval", help="score the pipeline against resolved questions")
    p_eval.add_argument("--dataset", default="evals/questions.json")
    p_eval.set_defaults(func=cmd_eval)

    p_sim = sub.add_parser("simulate", help="agent-based belief simulation")
    p_sim.add_argument("--agents", type=int, default=40)
    p_sim.add_argument("--truth", type=float, default=0.7)
    p_sim.add_argument("--rounds", type=int, default=60)
    p_sim.add_argument("--social-weight", type=float, default=0.3)
    p_sim.add_argument("--topology", default="dense", choices=["dense", "ring"])
    p_sim.add_argument("--noise", type=float, default=0.12)
    p_sim.add_argument("--seed", type=int, default=7)
    p_sim.set_defaults(func=cmd_simulate)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
