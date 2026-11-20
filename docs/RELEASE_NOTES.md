# Release preparation

This repository was prepared as an independent copy on 2026-11-20. Existing project and environment files were read only.

## Changes in the release copy

- Added a concise paper-based README, the teaser/framework figures, and workflow/environment/asset documentation.
- Exported the installed model environment to requirement.txt; replaced machine-local package references with version pins and -e . . The plural requirements.txt forwards to it; the upstream list is retained in docs/requirements.upstream.txt.
- Added a preview-first path configuration helper and a checkpoint location manifest. The archived training/inference logic and experiment identities are retained.
- Restricted editable package discovery to the model/training namespaces so that bundled simulators and alternative policy projects are not discovered as top-level packages.
- Changed one nested f-string quotation in robotwin/round11/tail_dashboard.py for Python 3.10 syntax compatibility.
- Replaced the empty upstream ACT SIM_TASK_CONFIGS.json placeholder with an empty JSON object; task-specific values still need to be supplied when using that alternative policy.
- Removed notebook execution outputs in the copied notebooks and resolved the selected RoboTwin source symlinks into ordinary files.
- Included the bundled RoboTwin C++/CUDA kernels, build manifests and Dockerfiles; native extensions were not compiled during publication.
- Added ignore rules for generated data, weights, caches and machine-local configuration.

These preparation changes do not modify model tensors, training objectives or benchmark results. Original upstream attribution and component license notices are retained.

## Verification

All Python source is parsed without importing training modules or writing bytecode. Shell scripts receive bash -n checks, JSON files are parsed, and the new path helper is exercised on disposable files (preview, relocation, repeated use and rejection of external symlinks). Files are scanned for common credential patterns and oversized artifacts.

The original source hashes and modification times are checked against a private export manifest; a second environment export is compared with the first. No training, GPU inference, benchmark evaluation or dependency installation is run during publication.
