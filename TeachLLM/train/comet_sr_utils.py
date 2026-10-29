from __future__ import annotations
import logging
import re
from typing import Any
logger = logging.getLogger(__name__)

def get_hf_comet_experiment(trainer: Any) -> Any:
    CometCallback: Any = None
    try:
        from transformers.integrations import CometCallback as _CC
        CometCallback = _CC
    except ImportError:
        try:
            from transformers.integrations.integration_utils import CometCallback as _CC2
            CometCallback = _CC2
        except ImportError:
            pass
    for cb in getattr(trainer, 'callback_handler', object()).callbacks:
        if CometCallback is not None and isinstance(cb, CometCallback):
            exp = getattr(cb, '_experiment', None)
            if exp is not None:
                return exp
        if cb.__class__.__name__ == 'CometCallback':
            exp = getattr(cb, '_experiment', None)
            if exp is not None:
                return exp
    return None

def comet_log_scalar(experiment: Any, name: str, value: float, *, step: int, epoch: float=0.0) -> None:
    if experiment is None:
        return
    try:
        lm = getattr(experiment, 'log_metric', None)
        if callable(lm):
            lm(name, float(value), step=int(step))
            return
    except Exception as exc:
        logger.debug('comet log_metric failed for %s: %s', name, exc)
    try:
        api = getattr(experiment, '__internal_api__', None)
        log_metrics = getattr(api, 'log_metrics__', None) if api is not None else None
        if callable(log_metrics):
            log_metrics({name: float(value)}, step=int(step), epoch=float(epoch), framework='transformers')
            return
    except Exception as exc:
        logger.warning('Comet SR: could not log %s=%s at step %s: %s', name, value, step, exc)

def _sanitize_task_name(raw_name: str, task_idx: int) -> str:
    task_name = re.sub('[^A-Za-z0-9_]+', '_', str(raw_name)).strip('_')
    if not task_name:
        task_name = f'task_{task_idx}'
    return task_name

def log_mappo_style_sr_to_comet(experiment: Any, rows: list[list[int]], task_names: tuple[str, ...], cum: dict[str, Any], *, step: int, epoch: float=0.0) -> None:
    if not rows or experiment is None:
        return
    finished_now = int(len(rows))
    solved_now = int(sum((int(r[2]) for r in rows)))
    cum['total_finished'] = int(cum.get('total_finished', 0)) + finished_now
    cum['total_issued'] = int(cum.get('total_issued', 0)) + finished_now
    cum['total_solved'] = int(cum.get('total_solved', 0)) + solved_now
    task_counts: dict[int, int] = cum.setdefault('task_counts', {})
    task_solved: dict[int, int] = cum.setdefault('task_solved', {})
    for _instr, task_idx, succ in rows:
        t = int(task_idx)
        task_counts[t] = int(task_counts.get(t, 0)) + 1
        task_solved[t] = int(task_solved.get(t, 0)) + int(succ)
    cumulative_total_solved = int(cum['total_solved'])
    cumulative_total_issued = int(cum['total_issued'])
    cumulative_total_finished = int(cum['total_finished'])
    cumulative_honest_micro = cumulative_total_solved / max(cumulative_total_issued, 1)
    comet_log_scalar(experiment, 'episode/Instructions/SR', float(cumulative_honest_micro), step=step, epoch=epoch)
    for task_idx in sorted({int(r[1]) for r in rows}):
        idx_count_cum = int(task_counts.get(task_idx, 0))
        idx_solved_cum = int(task_solved.get(task_idx, 0))
        idx_sr_cum = idx_solved_cum / max(idx_count_cum, 1)
        if task_names and 0 <= task_idx < len(task_names):
            raw_name = str(task_names[task_idx])
        else:
            raw_name = f'task_{task_idx}'
        tname = _sanitize_task_name(raw_name, task_idx)
        comet_log_scalar(experiment, f'instruction/sr_{tname}', float(idx_sr_cum), step=step, epoch=epoch)
        comet_log_scalar(experiment, f'instruction/count_{tname}', float(idx_count_cum), step=step, epoch=epoch)
    comet_log_scalar(experiment, 'instruction/sr_honest_micro', float(cumulative_honest_micro), step=step, epoch=epoch)
    comet_log_scalar(experiment, 'instruction/count_honest_micro', float(cumulative_total_issued), step=step, epoch=epoch)
    finished_only_sr = cumulative_total_solved / max(cumulative_total_finished, 1)
    comet_log_scalar(experiment, 'instruction/sr_honest_micro_finished_only', float(finished_only_sr), step=step, epoch=epoch)
    comet_log_scalar(experiment, 'instruction/count_honest_micro_finished_only', float(cumulative_total_finished), step=step, epoch=epoch)
    if task_names:
        task_srs = [float(task_solved.get(t, 0)) / max(float(task_counts.get(t, 0)), 1.0) for t in range(len(task_names))]
        honest_macro_sr = float(sum(task_srs) / max(len(task_srs), 1))
        num_tasks = len(task_names)
    elif task_counts:
        task_srs = [float(task_solved[t]) / max(float(task_counts[t]), 1.0) for t in sorted(task_counts.keys())]
        honest_macro_sr = float(sum(task_srs) / max(len(task_srs), 1))
        num_tasks = len(task_counts)
    else:
        honest_macro_sr = 0.0
        num_tasks = 0
    comet_log_scalar(experiment, 'instruction/sr_honest_macro', honest_macro_sr, step=step, epoch=epoch)
    comet_log_scalar(experiment, 'instruction/count_honest_macro', float(num_tasks), step=step, epoch=epoch)
    comet_log_scalar(experiment, 'sr_honest_macro', honest_macro_sr, step=step, epoch=epoch)
