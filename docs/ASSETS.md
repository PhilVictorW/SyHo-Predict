# Assets and checkpoints

This Git repository contains source, configuration, task descriptions, documentation and the two paper figures. It excludes model tensors, optimizer states, demonstrations, simulator meshes/textures, videos, logs and generated experiment outputs.

## External inputs

Set `asset_root` in `configs/paths.local.json` to the directory containing this layout:

```text
<asset_root>/
  results/Checkpoints/
    libero/lawam_libero_sft_release/
      final_model/pytorch_model.pt
      dataset_statistics.json
    robotwin/lawam_robotwin_sft_release/
      final_model/pytorch_model.pt
      dataset_statistics.json
    qwen3_weights/                          # configs + processor/tokenizer
  weights/dinov3-vitb16-pretrain-lvd1689m/   # backbone configuration
  dataset/
    libero_merged_no_noops_20hz/
    robotwin_merged/
    libero-plus/inference/LIBERO-plus/
```

LIBERO-Plus training uses the separately configured `plus_indexed_data` directory. The included dataset readers define the expected schema: `goal_local/data.py`, `libero-plus/runtime/indexed_future.py`, and `robotwin/data.py`.

Install the benchmark assets and provide valid LIBERO configuration files. The RoboTwin runtime needs its external assets/data as well as native SAPIEN dependencies. The modified reset simulator source is retained at:

`stage_III/robotwin/launchers/goal_local_gpu1_20261014_220941/simulator_reset_v2/`

Its original linked text/source files have been copied as ordinary files. Large external asset directories were not copied. Supply those assets and select the intended simulator root in the workflow configuration.

## Checkpoint archive

The owner's archive is [Phil-V/SyHo-Predict-models](https://huggingface.co/Phil-V/SyHo-Predict-models). It is currently private and requires permission; publishing this code does not change its visibility.

[checkpoint_layout.json](../configs/checkpoint_layout.json) maps the archive's individual files to either a repository-relative destination (`root: repo`) or an external asset-relative destination (`root: assets`). This is a location manifest, not a downloader. Keep processors, checkpoint completion markers, action semantics and baseline evaluation records together with the tensors.

| Benchmark | Selected Stage I | Selected Stage II / interaction | Stage III |
| --- | --- | --- | --- |
| LIBERO | teacher step 2000 | student step 4000 | frozen student + gating actor |
| LIBERO-Plus | teacher step 5000 | student step 4000 | frozen student + gating actor |
| RoboTwin, earlier branch | teacher step 1000 | student step 3000 | not the final interaction base |
| RoboTwin, final branch | SFT + round4/round6 adapters | round7 initialization + interaction student step 200 | interaction student + gating actor |

The archive retains selected models and lineage; it is not a full optimizer/RNG snapshot of every experiment. Archived configuration strings inside binary checkpoints are unchanged. The path helper only relocates text source/configuration, so exact resume checks may need a separately established compatible run identity. Never rewrite historical hash-checked results while presenting them as original evidence.

## Included source and exclusions

Core model and training code is under `starVLA/`, `latent_action_model/`, `goal_local/`, `robotwin/`, `libero-plus/` and `stage_III/`. The release also preserves ablations, deployment/evaluation adapters, task files, benchmark source and historical source snapshots.

The machine-specific project/environment files were only read during export. Publication files were created in an independent directory. The original research notes remain available for provenance and may reference local files, old run states or assets excluded from Git.

The manuscript also depicts real-robot experiments. This release documents the inspected simulation workflows; it does not assert that all real-robot deployment software or every experiment in the manuscript is bundled.
