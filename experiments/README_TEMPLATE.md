# Template for the README of an experiment folder

Every folder of `experiments/` holds a `README.md` written from this template
(issue #250), so that the folder reads as a short, standalone scientific
report: what the experiment asks, how it is simulated, what it found, and the
figures that show it. Copy the block below into `experiments/<name>/README.md`
and fill each section. The run commands, every option of the launcher and the
layout of the output folders stay in [`experiments/README.md`](README.md): the
README of a folder links to its section there instead of repeating it.

Conventions:

- `N` is the number of **system** qubits, the ancilla (reference qubit `R`)
  excluded (issue #235). A figure made before issue #235 labels its sizes in
  the old convention, `N + 1`; say so in its caption.
- Figures live in `experiments/<name>/figures/`, as PNG, and are linked with a
  relative path, `![...](figures/<file>.png)`. They are copies of the figures
  that the plot scripts of the folder write under `outputs/` (which git
  ignores); each caption names the script and the issue or PR that produced
  it, so that the figure can be regenerated.
- Each caption says what is on the axes, what the curves, colors and error
  bars are, and what the figure shows physically.
- Results quote numbers with the sizes, seeds and sample counts they were
  obtained with. A result that is not from code on `main` (a scratch run) is
  marked as such.

````markdown
# <Experiment title>

One paragraph: the question the experiment answers and how it relates to the
measurement-induced phase transition (MIPT) and the QELM.

## Experiment overview

| | |
| --- | --- |
| Script(s) | `experiments/<name>/<script>.py`, launcher `<script>.sh` |
| Config | `conf/<file>.yaml` |
| Physical model | Hamiltonian or circuit, boundary conditions |
| System sizes `N` | ... |
| Control parameters | measurement probability `p`, strength `gamma`, field `h`, ... |
| Initial state | ... |
| ML task | target, features, readout (ridge regression, K-NN, ...), score |
| Issues / PRs | #... |

## Methodology

Numbered steps of one sample or trajectory: state preparation, unitary
steps, measurements, how the features (`M`, `<Z_k>`, ...) and the target are
built, the train/test split and the score. Then how a sweep is run and
averaged (seeds, error bars).

How to run it: link to the section of `experiments/README.md`.

## Results and discussion

Bullet points of the main findings, each with its numbers and the figure that
shows it, and the caveats (finite size, convention of `N`, open questions).

## Figures

![Short alt text](figures/<file>.png)

**Figure 1. Title.** Axes, curves, colors, error bars, parameters. What the
figure shows and why it matters for the MIPT. Produced by
`experiments/<name>/<plot script>.py` (issue #...).
````
