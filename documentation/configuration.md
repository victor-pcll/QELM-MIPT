# Configuration

Scientific parameters and secrets live in different files.

## Hydra YAML

`main.py` loads `conf/config.yaml` with Hydra (`version_base="1.3"`). Command-line overrides are applied on top. `hydra.job.chdir` is `false`, so the process stays in the repository root.

`conf/config.yaml` selects the groups and holds the shared settings:

| Key | Role |
| --- | --- |
| `defaults` | `models: TFIM`, `training/measurements: projective`, `training/readout@single_observable: single`, `training/readout@double_observable: double`, `training/regression: ridge`, `evaluation: default` |
| `seed` | Integer passed to NumPy at startup. See [Code quality](code-quality.md). |
| `ancilla_charge_sector.N` | Total qubit count of the initial state, `${${models}.N}`: it follows the selected Hamiltonian. Read by `AncillaChargeSectorEncoding.from_config(cfg)` |
| `samples.L`, `samples.train_fraction`, `samples.train_size` | Number of samples and training split, `train_size = int(0.7 * L)`. Not read by the pipeline yet |
| `single_observable.observables`, `double_observable.pairs` | Features measured on the final state: `<O_i>` for each label (default `[Z]`), `<A_i B_j>` for each pair (default `[[Z, Z]]`). Several entries allowed; `[]` or a missing block gives no feature. Read by `SingleObservable.from_config(cfg)` / `DoubleObservable.from_config(cfg)` |
| `single_observable.N`, `double_observable.N` | Qubit count, `${${models}.N}`. `operators()` uses it when called without `N`; `operators(N=...)` overrides it |
| `projective_measurement.qubits` | Sites measured by `ProjectiveMeasurement`, in order; `null` (default) measures every system qubit `Nr .. N-1`. Read by `ProjectiveMeasurement.from_config(cfg)`, which also works without the block |
| `projective_measurement.Nr` | Ancilla qubits at the start of the chain, default `1` (the single ancilla of `AncillaChargeSectorEncoding`). With `qubits: null` they are never measured; an explicit `qubits` list ignores it. Without the key, `Nr = 0` |
| `projective_measurement.p` | Probability of measuring each target qubit, default `1.0`. `measure(rho)` uses it; `measure(rho, p=...)` overrides it; without the key, `p = 1.0`. It is the only measurement probability in the config: override it with `projective_measurement.p=0.5` |
| `checkpoint` | Where the run writes its `.npz`. See [Checkpoints and local data](checkpoints.md). |
| `hydra.run.dir` | `outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}` |

Group files:

| File | Contents |
| --- | --- |
| `conf/models/TFIM.yaml` | `models: TFIM`, the `TFIM` block (`N`, `J`, `h = 10 J`, `basis`, `RUNS` random Hamiltonians; `basis: X` gives `H = h Σ Z_i + Σ J_ij X_i X_j`, `basis: Z` the condensed matter convention `H = -h Σ X_i - Σ J_ij Z_i Z_j`, issue #173) and the evolution block `evolution` (`t = 10 / J`, `n_steps = d = 4 N`, `dt = t / d`) |
| `conf/models/haar_circuit.yaml` | `models: haar_circuit`, a `haar_circuit` block (`N`, `Nr`, `layers_per_step: 1`, `boundary: open`, `fixed_instance: false`): a brickwork of Haar-random two-qubit gates on the system qubits, the unitary dynamics of `experiments/QELM_M` in place of the TFIM (issue #171) |
| `conf/models/clifford_circuit.yaml` | `models: clifford_circuit`, the same block for a brickwork of uniformly random two-qubit Clifford gates (issue #171) |
| `conf/models/kitaev.yaml` | `models: kitaev`, a `kitaev` block with only `N`, and a fixed `evolution` block (`dt: 0.01`, `n_steps: 100`) |
| `conf/training/measurements/projective.yaml` | The `projective_measurement` block (`qubits: null`, `Nr: 1`, `p: 1.0`), placed by its `# @package projective_measurement` header |
| `conf/training/readout/single.yaml` | The `single_observable` block (`N`, `observables: [Z]`), placed by `training/readout@single_observable` in the defaults list |
| `conf/training/readout/double.yaml` | The `double_observable` block (`N`, `pairs: [[Z, Z]]`), placed by `training/readout@double_observable` in the defaults list |
| `conf/training/regression/ridge.yaml` | `name: ridge`, `alpha: 0.001` (ridge regularization, 1e-3 because 1.0 also shrinks the bias weight and costs R2; `fit(..., alpha=...)` overrides it), under `training` (`# @package training`) |
| `conf/training/regression/linear.yaml` | `name: linear`, under `training` (`# @package training`) |

The tree under `conf/` mirrors `src/QELM/`: `models/` for `QELM.models`, and `training/measurements/`, `training/readout/` and `training/regression/` for the `QELM.training` subpackages. The composed keys (`projective_measurement`, `single_observable`, `double_observable`, `training`) are the same as before the split, so the code reads them unchanged. Select a file with its group path: `models=kitaev`, `training/regression=linear`.
| `conf/evaluation/default.yaml` | `name: default`, a list of metric names, and a plot name |

`evolution.dt` and `evolution.n_steps` are recorded by `evolution_request` in the pipeline. `TFIM.RUNS` is bound to `model.RUNS` (not used by the pipeline yet), and `from_config` adds `evolution.dt` to the model params as `dt`.

## Derived values

Some values are computed from others with four resolvers registered by `import QELM` (`src/QELM/resolvers.py`): `${mul:a,b,...}`, `${sub:a,b}`, `${div:a,b}` and `${int:x}`. For example `h: ${mul:10.0,${TFIM.J}}`. A derived value follows its source, so `TFIM.J=2` gives `h=20`, `t=5` and a smaller `dt`, and `N=12` gives `n_steps=48`. It can still be overridden directly (`TFIM.h=3`, `evolution.dt=0.05`); the other derived values then keep their formula, so after `evolution.dt=0.05` the product `n_steps * dt` no longer equals `t`.

`N`, `J`, `dt`, and `seed` are config values. They are not constants in the Python package. Change them in YAML or with a CLI override. See [Running](running.md).

## `.env` secrets

API keys and machine-local paths belong in `.env`. That file is gitignored. Copy the example and fill it in locally:

```bash
cp .env.example .env
```

`.env.example` only lists `WANDB_API_KEY`. `main.py` calls `load_secrets()`, which runs `load_dotenv()`. A missing `.env` is ignored. The pipeline does not read the key and does not upload a run.

## Qubit counts: N, Nr, N_total

Three top-level keys name the qubit counts everywhere (issue #235, which replaces the convention of issue #88):

- `N`: the number of **system** qubits, the size of the studied system, ancilla **excluded**: the `N` of finite-size scaling and of the papers (their `L`). Each model file sets its default (`conf/models/TFIM.yaml`: 10); override it with `N=6`.
- `Nr`: the number of **ancilla** (reference) qubits, the first sites of the chain. Default 1 (`conf/config.yaml`), which matches `AncillaChargeSectorEncoding`; 0 in the configs without ancilla (`conf/learnability.yaml`, `conf/QELM_S.yaml`, `conf/double_descent.yaml`).
- `N_total`: the number of qubits of the **simulated state**, never set by hand: `N_total: ${int:${add:${N},${Nr}}}`, so `N_total = N + Nr` always holds. A density matrix is `2**N_total x 2**N_total`.

The model block (`TFIM.N`, the size of the Hamiltonian or circuit of the system) and the encoding (`ancilla_charge_sector.N`) interpolate `N`, the projective and weak measurements (`projective_measurement.Nr`) `Nr`, and the readouts, which act on the whole state, `N_total` (`single_observable.n_qubits`, `double_observable.n_qubits`). `load_config` and the experiment scripts call `QELM.check_qubit_counts(cfg)`, which returns `(N, Nr, N_total)` (a named tuple, so `check_qubit_counts(cfg).N` works too) and raises `ValueError` when a copy disagrees with its source, for example after `TFIM.N=6` instead of `N=6`, or after `N_total=7`. Scripts take the counts from it, never from a model or an encoding. In Python, `N`, `Nr` and `N_total` are also the attribute names (`Encoding.N`, `Encoding.N_total`, `BrickworkCircuit.N_total`, `ProjectiveMeasurement.Nr`), and `QELM.physics.qubit_counts(rho, Nr)` returns `(N, Nr, N_total)` for a density matrix. The low-level helpers of `QELM.physics` (`get_site_operator(op, site, N)`, the partial trace, the entropies) keep taking `N` as the number of qubits of the operator or state they act on, whatever their role.

### Runs written before issue #235

Before issue #235, `N` counted the ancilla too (`N = Ns + Nr`) and a top-level `Ns` held the system size. A run folder of that time is recognized by that `Ns` key in its `.hydra/config.yaml`. `QELM.utils.load_sweep`, behind every plot script, reads such a run in the new convention (`QELM.config.upgrade_legacy_counts`): its row `N` is its number of system qubits, so an old run named `N10_...` shows up as `N = 9`. The folder names, the `.npz` and the `metrics.json` of an old run are not rewritten: a script that reads the size from a folder name, or a figure already saved, still carries the old `N`. To compare an old sweep with a new one at the same system size, pass `N` one less than before (`N=9` instead of `N=10` with one ancilla), or regenerate the old sweep.
