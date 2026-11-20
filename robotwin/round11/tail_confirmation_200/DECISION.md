# Round11 continuation decision

The uniform and tail 200-step continuation adapters completed the same 128-pair held-out confirmation. Neither candidate is eligible for online rollout. Fresh and aggregate tail errors are statistically non-improving, while the `remaining=1` subgroup violates the non-regression gate for both candidates (especially gripper error).

Deployment decision: keep `action_adapter_200/step_000200` as the offline base/reference; do not promote either continuation and do not claim a new closed-loop success rate.

Next controlled experiment: freeze the base actor, diagnose terminal action/role drift, then run a small terminal-aware trust-region probe with an explicit output-drift constraint. Require the same fresh plus terminal held-out gates before any online rollout or long training.
