"""Route a Hydra config to model, training, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig

from qrc_env_template.evaluation import Evaluator, get_evaluator
from qrc_env_template.models import HamiltonianModel, get_model
from qrc_env_template.training import Trainer, get_trainer


@dataclass(frozen=True)
class PipelinePlan:
    """Classes and callables selected for one config."""

    sim_type: str
    model: type[HamiltonianModel]
    trainer: type[Trainer]
    evaluator: Evaluator


def resolve_pipeline(cfg: DictConfig) -> PipelinePlan:
    """Map ``sim_type``, ``training.name``, and ``evaluation.name`` to modules."""
    sim_type = str(cfg.sim_type)
    return PipelinePlan(
        sim_type=sim_type,
        model=get_model(sim_type),
        trainer=get_trainer(str(cfg.training.name)),
        evaluator=get_evaluator(str(cfg.evaluation.name)),
    )


def run_pipeline(cfg: DictConfig) -> str:
    """Build the selected blocks and return a scaffold status line.

    ``dt``, ``n_steps``, and simulation parameters are read from ``cfg``.
    """
    plan = resolve_pipeline(cfg)
    model = plan.model.from_config(cfg)
    trainer = plan.trainer.from_config(cfg)
    evolution = model.evolve(dt=float(cfg.test.dt), n_steps=int(cfg.test.n_steps))
    training_status = trainer.fit(features=None, targets=None)
    report = plan.evaluator(evolution=evolution, training_status=training_status, cfg=cfg)
    return (
        f"model={model.name} training={trainer.name} "
        f"dt={evolution['dt']} n_steps={evolution['n_steps']} {report}"
    )


__all__ = ["PipelinePlan", "resolve_pipeline", "run_pipeline"]
