"""Unit tests for TransverseFieldIsingModel and HamiltonianModel base class."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from QELM.models.base import EvolutionState, HamiltonianModel, evolution_request
from QELM.models.TFIM import TransverseFieldIsingModel
from sparse_checks import assert_sparse_hermitian

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


@pytest.mark.parametrize("qubit_counts", [2, 3, 4])
def test_hamiltonian_shape_and_dimension(qubit_counts: int):
    """Vérifie que la matrice H a bien la forme (2^N, 2^N)."""
    params = {"N": qubit_counts, "J": 1.0, "h": 0.5}
    model = TransverseFieldIsingModel(params)

    expected_dim = 2**qubit_counts
    assert model.dim == expected_dim

    H = model.hamiltonian()
    assert H.shape == (expected_dim, expected_dim)
    assert H.dtype == np.complex128 or H.dtype == complex


def test_hamiltonian_hermiticity(
    ising_model: TransverseFieldIsingModel,
):
    """Vérifie la propriété physique fondamentale : H est Hermitien (H = H^dagger)."""
    # Vérification stricte sans copie dense (voir tests/sparse_checks.py)
    assert_sparse_hermitian(ising_model.hamiltonian(), atol=1e-12)


def test_hamiltonian_determinism_and_fixed_disorder(
    ising_model: TransverseFieldIsingModel,
):
    """Vérifie que deux appels successifs renvoient EXACTEMENT la même matrice H.

    (S'assure que le désordre J_ij n'est pas régénéré au hasard à chaque appel).
    """
    H_first = ising_model.hamiltonian()
    H_second = ising_model.hamiltonian()

    np.testing.assert_array_equal(H_first.toarray(), H_second.toarray())


# =====================================================================
# 3. TESTS DE L'INTERFACE HYDRA / OMEGACONF (from_config)
# =====================================================================


def test_from_config_instantiation():
    """Vérifie que le constructeur alternatif from_config fonctionne avec OmegaConf."""
    cfg = OmegaConf.create({"TFIM": {"N": 4, "J": 0.8, "h": 0.2}})

    model = TransverseFieldIsingModel.from_config(cfg)
    assert model.N == 4
    assert model.dim == 16
    assert model.J == 0.8


def test_from_config_invalid_type_raises_error():
    """Vérifie qu'une erreur de type est levée si la sous-clé n'est pas un mapping."""
    cfg = OmegaConf.create({"TFIM": "not_a_dictionary"})

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


def test_evolution_request_scaffold():
    """``evolution_request`` enregistre dt et n_steps sans intégrer."""
    base_model = HamiltonianModel({"N": 2})
    state: EvolutionState = evolution_request(base_model, dt=0.1, n_steps=10)

    assert state["dt"] == 0.1
    assert state["n_steps"] == 10
    assert state["model"] == "base"
    assert state["status"] == "scaffold"


def test_tfim_yaml_builds_model_with_field():
    """Le fichier conf/models/TFIM.yaml construit un TFIM avec ``h``."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from QELM.models import get_model

    GlobalHydra.instance().clear()
    conf = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(version_base="1.3", config_dir=str(conf)):
        cfg = compose(config_name="config", overrides=["N=3", "TFIM.h=0.7"])
    GlobalHydra.instance().clear()

    model = get_model(str(cfg.models)).from_config(cfg)
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


# =====================================================================
# 5. PARAMÈTRE ``basis`` (issue #173)
# =====================================================================


def _dense_tfim(
    h: float, J_matrix: np.ndarray, field: np.ndarray, coupling: np.ndarray
) -> np.ndarray:
    """``h * sum_i field_i + sum_{i<j} J_ij coupling_i coupling_j``, dense."""
    from QELM.physics.operators import get_site_operator

    N = J_matrix.shape[0]
    H = sum(h * get_site_operator(field, i, N) for i in range(N))
    for i in range(N):
        for j in range(i + 1, N):
            H = H + J_matrix[i, j] * (
                get_site_operator(coupling, i, N) @ get_site_operator(coupling, j, N)
            )
    return np.asarray(H)


def test_default_basis_is_x_and_unchanged():
    """Sans ``basis``, H est la convention X : h sum Z_i + sum J_ij X_i X_j."""
    from QELM.physics.operators import X, Z

    np.random.seed(3)
    model = TransverseFieldIsingModel({"N": 4, "J": 1.0, "h": 0.7})
    assert model.basis == "X"
    expected = _dense_tfim(0.7, model.J_matrix, field=Z, coupling=X)
    np.testing.assert_allclose(model.hamiltonian().toarray(), expected, atol=1e-14)

    np.random.seed(3)
    explicit = TransverseFieldIsingModel({"N": 4, "J": 1.0, "h": 0.7, "basis": "X"})
    np.testing.assert_array_equal(
        explicit.hamiltonian().toarray(), model.hamiltonian().toarray()
    )


def test_basis_z_is_condensed_matter_convention():
    """``basis = "Z"`` : H = -h sum X_i - sum J_ij Z_i Z_j, hermitien."""
    from QELM.physics.operators import X, Z

    np.random.seed(5)
    model = TransverseFieldIsingModel({"N": 4, "J": 1.0, "h": 0.7, "basis": "Z"})
    H = model.hamiltonian()
    expected = _dense_tfim(-0.7, -model.J_matrix, field=X, coupling=Z)
    np.testing.assert_allclose(H.toarray(), expected, atol=1e-14)
    assert_sparse_hermitian(H, atol=1e-12)


def test_bases_are_related_by_hadamard_and_sign_flip():
    """H_Z = -(H^{⊗N}) H_X (H^{⊗N}) avec le même désordre : spectres opposés."""
    N = 3
    np.random.seed(7)
    model_x = TransverseFieldIsingModel({"N": N, "J": 1.0, "h": 0.4})
    np.random.seed(7)
    model_z = TransverseFieldIsingModel({"N": N, "J": 1.0, "h": 0.4, "basis": "Z"})
    np.testing.assert_array_equal(model_x.J_matrix, model_z.J_matrix)

    hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    W = hadamard
    for _ in range(N - 1):
        W = np.kron(W, hadamard)
    H_x = model_x.hamiltonian().toarray()
    H_z = model_z.hamiltonian().toarray()
    np.testing.assert_allclose(H_z, -(W @ H_x @ W), atol=1e-13)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(H_z), np.sort(-np.linalg.eigvalsh(H_x)), atol=1e-12
    )


def test_invalid_basis_raises():
    """Une base autre que X ou Z est refusée."""
    with pytest.raises(ValueError, match="basis"):
        TransverseFieldIsingModel({"N": 2, "J": 1.0, "h": 0.5, "basis": "Y"})


def test_tfim_yaml_basis_default_and_override():
    """``conf/models/TFIM.yaml`` vaut ``basis: X`` ; ``TFIM.basis=Z`` le change."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from QELM.models import get_model

    conf = Path(__file__).resolve().parents[1] / "conf"
    for overrides, basis in ((["N=3"], "X"), (["N=3", "TFIM.basis=Z"], "Z")):
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3", config_dir=str(conf)):
            cfg = compose(config_name="config", overrides=overrides)
        GlobalHydra.instance().clear()
        model = get_model(str(cfg.models)).from_config(cfg)
        assert model.basis == basis


@pytest.mark.parametrize("N", [1, 2, 3, 5])
@pytest.mark.parametrize("h", [0.0, 0.37, 2.0])
def test_hamiltonian_matches_the_kronecker_products(N: int, h: float) -> None:
    """The bitwise H of both bases equals the Kronecker products (issue #201)."""
    from QELM.physics.operators import X, Z, get_site_operator

    np.random.seed(7)
    model = TransverseFieldIsingModel({"N": N, "J": 1.0, "h": h})
    expected = sum(h * get_site_operator(Z, i, N) for i in range(N)) + sum(
        model.J_matrix[i, j] * get_site_operator(X, i, N) @ get_site_operator(X, j, N)
        for i in range(N)
        for j in range(i + 1, N)
    )
    np.testing.assert_allclose(model.hamiltonian().toarray(), expected, atol=1e-12)
    np.random.seed(7)
    model_z = TransverseFieldIsingModel({"N": N, "J": 1.0, "h": h, "basis": "Z"})
    expected_z = -sum(h * get_site_operator(X, i, N) for i in range(N)) - sum(
        model_z.J_matrix[i, j] * get_site_operator(Z, i, N) @ get_site_operator(Z, j, N)
        for i in range(N)
        for j in range(i + 1, N)
    )
    np.testing.assert_allclose(model_z.hamiltonian().toarray(), expected_z, atol=1e-12)
