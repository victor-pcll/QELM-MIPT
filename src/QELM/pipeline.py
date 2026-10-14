"""Route a Hydra config to model, training, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig

from QELM.checkpoint import save_checkpoint
from QELM.evaluation import Evaluator, get_evaluator
from QELM.models import HamiltonianModel, get_model
from QELM.models.base import evolution_request
from QELM.training import Trainer, get_trainer


@dataclass(frozen=True)
class PipelinePlan:
    """Classes and callables selected for one config.

    Attributes:
        sim_type: Config key, such as ``"TFIM"`` or ``"kitaev"``.
        model: Hamiltonian class for that simulation.
        trainer: Training class named by ``cfg.training.name``.
        evaluator: Callable named by ``cfg.evaluation.name``.
    """

    sim_type: str
    model: type[HamiltonianModel]
    trainer: type[Trainer]
    evaluator: Evaluator


def resolve_pipeline(cfg: DictConfig) -> PipelinePlan:
    """Map config names to a model, trainer, and evaluator.

    Args:
        cfg: Composed Hydra config. ``cfg.sim_type``, ``cfg.training.name``,
            and ``cfg.evaluation.name`` are scalar strings.

    Returns:
        The selected classes and evaluator. Nothing is instantiated.
    """
    sim_type = str(cfg.sim_type)
    return PipelinePlan(
        sim_type=sim_type,
        model=get_model(sim_type),
        trainer=get_trainer(str(cfg.training.name)),
        evaluator=get_evaluator(str(cfg.evaluation.name)),
    )


def run_pipeline(cfg: DictConfig) -> str:
    """Build the selected blocks and return a scaffold status line.

    Args:
        cfg: Composed Hydra config. ``cfg.test.dt`` is a scalar step size
            and ``cfg.test.n_steps`` is a scalar step count. ``N``, ``J``,
            and ``learning_rate`` stay on the simulation block.

    Returns:
        One status line naming the model, trainer, ``dt``, ``n_steps``,
        the evaluator report, and the checkpoint path. No trajectory
        tensor is allocated. The checkpoint directory comes from
        ``cfg.checkpoint``.
    """
    plan = resolve_pipeline(cfg)
    model = plan.model.from_config(cfg)
    trainer = plan.trainer.from_config(cfg)
    evolution = evolution_request(
        model, dt=float(cfg.test.dt), n_steps=int(cfg.test.n_steps)
    )
    training_status = trainer.fit(features=None, targets=None)
    report = plan.evaluator(
        evolution=evolution,
        training_status=training_status,
        cfg=cfg,
    )
    saved = save_checkpoint(
        cfg,
        model=model,
        trainer=trainer,
        evolution=evolution,
        training_status=training_status,
    )
    return (
        f"model={model.name} training={trainer.name} "
        f"dt={evolution['dt']} n_steps={evolution['n_steps']} {report} "
        f"checkpoint={saved}"
    )


__all__ = ["PipelinePlan", "resolve_pipeline", "run_pipeline"]
