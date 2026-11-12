"""Command line entry point: `polycast forecast | serve | eval | simulate`."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime, timedelta

from .audit import ForecastStore
from .corpus import build_retriever
from .evaluation import load_resolved, recalibrate, score_resolved
from .pipeline import ForecastPipeline
from .providers import MockProvider, OpenAICompatibleProvider
from .schemas import Horizon, Question
from .simulation import run_simulation


def _pipeline(args) -> ForecastPipeline:
    corpus = getattr(args, "corpus", None) or os.getenv("POLYCAST_CORPUS")
    db = getattr(args, "db", None) or os.getenv("POLYCAST_DB", "polycast.db")
    audit_dir = getattr(args, "audit_dir", None) or os.getenv("POLYCAST_AUDIT_DIR", "runs")
    model = getattr(args, "model", None) or os.getenv("POLYCAST_MODEL")

    if model:
        provider = OpenAICompatibleProvider(
            model=model,
            api_key=os.getenv("POLYCAST_API_KEY"),
            base_url=os.getenv("POLYCAST_BASE_URL", "https://api.openai.com/v1"),
        )
    else:
        provider = MockProvider()
    return ForecastPipeline(
        provider=provider,
        retriever=build_retriever(corpus),
        store=ForecastStore(db),
        audit_dir=audit_dir,
    )


def cmd_forecast(args) -> int:
    resolve_by = (
        datetime.fromisoformat(args.resolve_by)
        if args.resolve_by
        else datetime.now(UTC) + timedelta(days=45)
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


ENV_BY_OPTION = {
    "corpus": "POLYCAST_CORPUS",
    "db": "POLYCAST_DB",
    "audit_dir": "POLYCAST_AUDIT_DIR",
    "model": "POLYCAST_MODEL",
}


def export_env(args) -> None:
    """Publish CLI options as the environment variables the app reads.

    The FastAPI application builds its pipeline from the environment, so the
    `serve` command has to hand its options over that way. Without this the
    flags are accepted and then silently ignored.
    """
    for option, variable in ENV_BY_OPTION.items():
        value = getattr(args, option, None)
        if value:
            os.environ[variable] = str(value)


def cmd_serve(args) -> int:
    import uvicorn

    export_env(args)
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
    # Shared options are attached to the top level parser and to every
    # subcommand, so `polycast --db x.db forecast` and `polycast forecast
    # --db x.db` both work. SUPPRESS keeps a subcommand from overwriting a
    # value that was given before the subcommand name.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus", default=argparse.SUPPRESS, help="path to corpus json")
    common.add_argument("--db", default=argparse.SUPPRESS, help="sqlite path")
    common.add_argument("--audit-dir", default=argparse.SUPPRESS, help="audit trace directory")
    common.add_argument("--model", default=argparse.SUPPRESS, help="model id for a hosted provider")

    parser = argparse.ArgumentParser(
        prog="polycast", description=__doc__, parents=[common]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_forecast = sub.add_parser("forecast", parents=[common], help="run one forecast")
    p_forecast.add_argument("--text", required=True)
    p_forecast.add_argument("--criteria", required=True)
    p_forecast.add_argument("--resolve-by", default=None)
    p_forecast.add_argument("--horizon", default="months", choices=[h.value for h in Horizon])
    p_forecast.add_argument("--reference-class", default=None)
    p_forecast.add_argument("--json", action="store_true")
    p_forecast.set_defaults(func=cmd_forecast)

    p_serve = sub.add_parser("serve", parents=[common], help="run the HTTP API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_eval = sub.add_parser(
        "eval", parents=[common], help="score the pipeline against resolved questions"
    )
    p_eval.add_argument("--dataset", default="evals/questions.json")
    p_eval.set_defaults(func=cmd_eval)

    p_sim = sub.add_parser("simulate", parents=[common], help="agent-based belief simulation")
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
