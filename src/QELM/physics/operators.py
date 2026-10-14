from __future__ import annotations

import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def get_site_operator(op: np.ndarray, site: int, N: int) -> np.ndarray:
  """Construit un opérateur local à 1 qubit agissant sur le qubit 'site' d'une chaîne à N qubits.

  Args:
      op: Matrice 2x2 (ex: X, Y, Z).
      site: Indice du qubit cible (0 à N-1).
      N: Nombre total de qubits dans le système.

  Returns:
      Matrice de taille (2^N, 2^N).
  """
  ops = [I] * N
  ops[site] = op
  res = ops[0]
  for o in ops[1:]:
    res = np.kron(res, o)
  return res