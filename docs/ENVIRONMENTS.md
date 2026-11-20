# Environments

## Model environment

`requirement.txt` was exported read-only from the model environment on 2026-11-20 using `pip freeze --all` and installed distribution metadata:

| Component | Exported version |
| --- | --- |
| Python | 3.10.21 |
| PyTorch | 2.6.0 |
| Transformers | 5.2.0 |
| FlashAttention | 2.7.4.post1 |

The file pins 140 installed distributions and installs this checkout with `-e .`. Three machine-specific references were normalized: the local FlashAttention wheel and pip build directory became version pins, and the original editable LaWAM checkout became `-e .`. `requirements.txt` forwards to this export; the original upstream dependency list is retained as `docs/requirements.upstream.txt`.

For a new machine, create a fresh Python 3.10 environment, then install from the repository root:

```bash
conda create -n syho-model python=3.10
conda activate syho-model
python -m pip install -r requirement.txt
```

FlashAttention must have a wheel compatible with Python, PyTorch, CUDA and the platform, or be built with a matching CUDA toolkit. The export records Python packages, not the NVIDIA driver, compiler, OS libraries, or a complete Conda image. An exact clean-machine installation has not been tested.

## Separate simulator processes

Training and policy inference use the model environment. Rollout workers run in separate environments to preserve benchmark-specific dependencies:

| Workload | Original environment name | Recorded simulator stack |
| --- | --- | --- |
| LIBERO Stage I/II evaluation and Stage III final evaluation | LIBERO4LaWAM | MuJoCo 3.3.2, robosuite 1.4.0 |
| LIBERO Stage III online training rollouts | LIBERO | MuJoCo 3.9.0, robosuite 1.4.0 |
| LIBERO-Plus rollouts | libero-plus | MuJoCo 3.3.2, robosuite 1.4.0 |
| RoboTwin rollouts | RoboTwin4LaWAM | SAPIEN 3.0.0b1 |

These are the installed versions observed during export, not a guarantee of all historical runs' versions. The simulator environments are **not** included in the model environment export. Their original setup files remain in the bundled benchmark source; install benchmark assets and native rendering dependencies separately.

## Configure a new checkout

Copy `configs/paths.example.json` to `configs/paths.local.json`, edit every location, then run `scripts/configure_paths.py` with that file. Its default mode previews changes; `--apply` rewrites only the copied source/configuration files listed in `configs/release_sources.json`. It never edits external environments or datasets.

The original absolute paths are retained in the archived source for provenance. The helper maps the historical main/sibling checkouts to this checkout, the original asset root to your data/weights directory, and each interpreter to its separate environment. Use a fresh clone to change to another path configuration. Binary checkpoint metadata is not rewritten: archived resume and adapter identity checks may require additional configuration matching; newly trained checkpoints inherit the configured paths.

Some historical LIBERO-Plus launchers refer to a local ImageMagick sysroot and a CUDA installation. Supply compatible native libraries or adapt those environment variables. GPU IDs, worker counts, dates, run names and resume rules remain experimental settings; review them before running a historical launcher.
