> Superseded on 2026-10-08: use CANDIDATE_PROTOCOL.md and candidate_online/. Historical original results are reused; only compatible candidates are now scheduled on GPU0/GPU3. Earlier paired-run artifacts remain in online/.

# Round 9: matched closed-loop deployment pilot

Predeclared before online outcomes: click_bell, handover_block, lift_pot, stack_blocks_two; demo_clean and demo_randomized; 5 episodes per task/mode/arm, 80 episodes total. Task selection covers pressing, handoff, bimanual carry, and stacking. One simulator slot. Task/mode arm order alternates. Use seen instructions as configured by the original deployment YAML, starting seed 0 (simulator candidate seeds start at 100000); verify accepted seeds and instructions match after each pair.

Original arm: original LaWAM checkout with released RoboTwin weights, native BF16 deployment. Compatible arm: round7 fixed_queries initialization from candidate_recipe.json, including round4/round6 repaired weights and frozen zero relevance-gate scales. This does NOT use the round8 200-step updates. It retains original LaWM predicted-future conditioning and original queries; it does not establish the pure aba2 ablation.

Both use current observations only, no environment state or real future frames. Common action client: 256x256 main+two wrist views, original normalization statistics, clipping/gripper convention/quaternion handling, horizon 36, replan every 36 steps, 30 Hz, no action ensemble, 10 flow steps. Per-query Gaussian noise seed hashes task, mode, accepted episode seed, query index. Original BF16 sampler and compatible BF16 Gaussian draws use the same seeded values; compatible keeps its previously validated FP32 parameters/Euler state with BF16 autocast. This is a deployment-level comparison, not an isolated precision experiment.

Startup checks: verify original code path, native output shape [1,36,32] (original, existing client truncates unused channels) or [1,36,16] (compatible), effective action shape [1,36,16], finite actions, repeatability under the same seed, and compatible wrapper replay of the fixed offline validation frame. Client checks the server arm and horizon in metadata. No real future encoder may run in compatible deployment.

Use GPU0 only because earlier BF16 repeatability probes on GPU5 were unstable. Each policy allocator is capped at 25% GPU memory; low CPU/IO priority. Other processes and old Stage II pause settings are not changed. Each job skips after 2400 seconds without a completed episode; incomplete jobs are reported separately. Cleanup targets only process groups started by this queue. No videos are recorded.

Report per-task/mode counts and matched episode outcomes; paired seed/instruction mismatch stops the queue. 40 paired episodes are a screening pilot, not a 50-task benchmark or proof of an overall improvement.

Dashboard: `tmux attach -t robotwin_round9_progress`
Queue logs: `round9/queue.log`; results and status: `round9/online/`.

Instruction-pairing correction during first arm: native template generation uses global Python random despite seeded final selection. Repeated same-seed first-frame images and policy noise were identical but language text differed. Subsequent arm therefore generates its native instruction as before, then replays the first arm's recorded instruction for the same episode seed. The first arm's instructions are selected before outcomes; task/seed/weight selection is unchanged. Verify exact text equality in final pairing checks.

Round10 expands the same paired protocol to all 50 DEFAULT_TASKS, both modes, and 10 episodes per task/mode/arm (1000 episodes total).
