"""Unit tests for TransverseFieldIsingModel and HamiltonianModel base class."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from QELM.models.base import EvolutionState, HamiltonianModel
from QELM.models.TFIM import TransverseFieldIsingModel

# =====================================================================
# 1. FIXTURES (Données et modèles réutilisables)
# =====================================================================


@pytest.fixture
def default_params() -> dict[str, float]:
    """Paramètres d'entrée standard pour les tests."""
    return {"N": 3, "J": 1.0, "h": 0.5}


@pytest.fixture
def ising_model(default_params: dict[str, float]) -> TransverseFieldIsingModel:
    """Instance initialisée du modèle TFIM sur 3 qubits."""
    return TransverseFieldIsingModel(default_params)


# =====================================================================
# 2. TESTS DES PROPRIÉTÉS STRUCTURALES ET PHYSIQUES
# =====================================================================


@pytest.mark.parametrize("n_qubits", [2, 3, 4])
def test_hamiltonian_shape_and_dimension(n_qubits: int):
    """Vérifie que la matrice H a bien la forme (2^N, 2^N)."""
    params = {"N": n_qubits, "J": 1.0, "h": 0.5}
    model = TransverseFieldIsingModel(params)

    expected_dim = 2**n_qubits
    assert model.dim == expected_dim

    H = model.hamiltonian()
    assert H.shape == (expected_dim, expected_dim)
    assert H.dtype == np.complex128 or H.dtype == complex


def test_hamiltonian_hermiticity(
    ising_model: TransverseFieldIsingModel,
):
    """Vérifie la propriété physique fondamentale : H est Hermitien (H = H^dagger)."""
    H = ising_model.hamiltonian()
    H_dagger = H.conj().T

    # Vérification numérique stricte sur matrices complexes
    np.testing.assert_allclose(H, H_dagger, atol=1e-12)


def test_hamiltonian_determinism_and_fixed_disorder(
    ising_model: TransverseFieldIsingModel,
):
    """Vérifie que deux appels successifs renvoient EXACTEMENT la même matrice H.

    (S'assure que le désordre J_ij n'est pas régénéré au hasard à chaque appel).
    """
    H_first = ising_model.hamiltonian()
    H_second = ising_model.hamiltonian()

    np.testing.assert_array_equal(H_first, H_second)


# =====================================================================
# 3. TESTS DE L'INTERFACE HYDRA / OMEGACONF (from_config)
# =====================================================================


def test_from_config_instantiation():
    """Vérifie que le constructeur alternatif from_config fonctionne avec OmegaConf."""
    cfg = OmegaConf.create({"ising": {"N": 4, "J": 0.8, "h": 0.2}})

    model = TransverseFieldIsingModel.from_config(cfg)
    assert model.N == 4
    assert model.dim == 16
    assert model.J == 0.8


def test_from_config_invalid_type_raises_error():
    """Vérifie qu'une erreur de type est levée si la sous-clé n'est pas un mapping."""
    cfg = OmegaConf.create({"ising": "not_a_dictionary"})

    with pytest.raises(TypeError, match="config must be a mapping"):
        TransverseFieldIsingModel.from_config(cfg)


# =====================================================================
# 4. TESTS DE LA CLASSE DE BASE (HamiltonianModel)
# =====================================================================


def test_base_class_raises_not_implemented():
    """Vérifie que la classe abstraite de base lève bien NotImplementedError."""
    base_model = HamiltonianModel({"N": 2})

    with pytest.raises(NotImplementedError):
        base_model.hamiltonian()


def test_base_class_evolve_scaffold():
    """Vérifie le comportement de la méthode stub evolve()."""
    base_model = HamiltonianModel({"N": 2})
    state: EvolutionState = base_model.evolve(dt=0.1, n_steps=10)

    assert state["dt"] == 0.1
    assert state["n_steps"] == 10
    assert state["status"] == "scaffold"


def test_ising_yaml_builds_tfim_with_field():
    """Le fichier conf/sim_type/ising.yaml construit un TFIM avec ``h``."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from QELM.models import get_model

    GlobalHydra.instance().clear()
    conf = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(version_base="1.3", config_dir=str(conf)):
        cfg = compose(config_name="config", overrides=["ising.N=3", "ising.h=0.7"])
    GlobalHydra.instance().clear()

    model = get_model(str(cfg.sim_type)).from_config(cfg)
    assert isinstance(model, TransverseFieldIsingModel)
    assert model.h == 0.7
    assert model.hamiltonian().shape == (8, 8)


def test_disorder_is_reproducible_with_seed():
    """Même seed, même désordre J_ij."""
    np.random.seed(0)
    first = TransverseFieldIsingModel({"N": 3, "J": 1.0, "h": 0.5}).J_matrix
    np.random.seed(0)
    second = TransverseFieldIsingModel({"N": 3, "J": 1.0, "h": 0.5}).J_matrix
    np.testing.assert_array_equal(first, second)
