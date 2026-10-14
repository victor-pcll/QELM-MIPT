"""Unit tests for TransverseFieldIsingModel and HamiltonianModel base class."""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf
import pytest
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