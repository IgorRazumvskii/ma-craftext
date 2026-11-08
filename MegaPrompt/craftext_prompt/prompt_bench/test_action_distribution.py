"""Prompt bench: action parse rate and distribution for a MegaPrompt template."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

MEGAPROMPTS_ROOT = Path(__file__).resolve().parents[2]
if str(MEGAPROMPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGAPROMPTS_ROOT))

try:
    from .llm import DEFAULT_ALLOWED_ACTIONS, LLMConfig, extract_action, predict_action_from_prompt
    from .trajectory_fixture import extract_state_from_entry, load_trajectory_entry
    from .test_sr import DEFAULT_STEP_IDX, render_prompt
except Exception:  # pragma: no cover
    from llm import DEFAULT_ALLOWED_ACTIONS, LLMConfig, extract_action, predict_action_from_prompt
    from trajectory_fixture import extract_state_from_entry, load_trajectory_entry
    from test_sr import DEFAULT_STEP_IDX, render_prompt

DEFAULT_RENDER_CONFIG = (
    Path(__file__).resolve().parents[1] / "templates" / "no_dialog_but_goal" / "craftext.yaml"
)

NAV_ACTIONS = frozenset({"NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO"})
CARDINAL_NAV = frozenset({"UP", "DOWN", "LEFT", "RIGHT"})


def classify_action(raw: str, parsed: str | None) -> str:
    if parsed is None:
        return "PARSE_FAIL"
    upper = parsed.upper().strip()
    if upper in NAV_ACTIONS:
        return upper
    if upper.startswith("DO"):
        return "DO"
    return parsed


def test(
    *,
    step_idx: int = DEFAULT_STEP_IDX,
    run_count: int = 20,
    render_config_path: str | Path = DEFAULT_RENDER_CONFIG,
    llm_cfg: LLMConfig | None = None,
) -> dict[str, Any]:
    entry = load_trajectory_entry(step_idx=step_idx)
    state = extract_state_from_entry(entry)
    act_prompt = render_prompt(entry=entry, state=state, render_config_path=render_config_path)

    cfg_llm = llm_cfg or LLMConfig(
        model_name_or_path=os.environ.get("CRAFTEXT_LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
        max_new_tokens=int(os.environ.get("CRAFTEXT_LLM_MAX_NEW_TOKENS", "256")),
        temperature=1.0,
        top_p=0.9,
    )

    labels: list[str] = []
    errors = 0
    for _ in range(run_count):
        try:
            action, raw = predict_action_from_prompt(act_prompt, allow_ask_operator=False, cfg=cfg_llm)
            labels.append(classify_action(raw, action or extract_action(raw)))
        except Exception as exc:
            errors += 1
            labels.append(f"ERROR:{exc}")

    counter = Counter(labels)
    parse_fail = counter.get("PARSE_FAIL", 0) + sum(1 for k in counter if str(k).startswith("ERROR:"))
    nav_hits = sum(counter.get(a, 0) for a in NAV_ACTIONS)
    parse_ok = run_count - parse_fail
    action_distribution = dict(counter)
    distinct_nav_actions = sum(
        1 for action in CARDINAL_NAV if action_distribution.get(action, 0) > 0
    )

    result = {
        "render_config": str(render_config_path),
        "run_count": run_count,
        "parse_fail_count": parse_fail,
        "parse_fail_rate": parse_fail / run_count,
        "distinct_nav_actions": distinct_nav_actions,
        "nav_action_rate": nav_hits / max(parse_ok, 1),
        "action_distribution": action_distribution,
        "prompt_snippet": act_prompt[:1200],
    }

    print(json.dumps({k: v for k, v in result.items() if k != "prompt_snippet"}, indent=2))
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Action distribution bench for MegaPrompt craftext templates.")
    parser.add_argument("--render-config", type=str, default=str(DEFAULT_RENDER_CONFIG))
    parser.add_argument("--step-idx", type=int, default=DEFAULT_STEP_IDX)
    parser.add_argument("--run-count", type=int, default=20)
    parser.add_argument(
        "--llm-model",
        type=str,
        default=os.environ.get("CRAFTEXT_LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
    )
    parser.add_argument(
        "--llm-max-new-tokens",
        type=int,
        default=int(os.environ.get("CRAFTEXT_LLM_MAX_NEW_TOKENS", "256")),
    )
    parser.add_argument("--llm-temperature", type=float, default=1.0)
    parser.add_argument("--llm-top-p", type=float, default=0.9)
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    llm_cfg = LLMConfig(
        model_name_or_path=args.llm_model,
        max_new_tokens=args.llm_max_new_tokens,
        temperature=args.llm_temperature,
        top_p=args.llm_top_p,
    )
    out = test(
        step_idx=args.step_idx,
        run_count=args.run_count,
        render_config_path=args.render_config,
        llm_cfg=llm_cfg,
    )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
