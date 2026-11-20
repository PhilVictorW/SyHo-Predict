# abstract

Existing robotic manipulation models leverage future-state prediction to guide action generation. Yet insufficient modeling of multi-horizon future representations limits their ability to balance precise local execution with long-horizon task progression. 

We observe that near-term futures capture fine-grained local state changes, while distant futures characterize broader trends in task evolution, providing complementary guidance for action generation. Motivated by this insight, we propose **ReCo-WAM**, a role-specialized world-action model with reward-guided cross-horizon interaction. Specifically, we disentangle future information into near-term state changes and long-horizon task evolution, learn their respective posterior teacher representations, and transfer them to a prior student model through multi-layer distillation. The resulting representations jointly guide action generation through dedicated conditioning pathways for local execution and long-horizon planning. Furthermore, we introduce a reward-guided cross-horizon interaction mechanism that employs continuous gating and group-relative policy optimization to adaptively regulate the influence of long-horizon representations on local state predictions, enabling task progression cues to selectively refine local decisions. 



Extensive experiments on [benchmarks] demonstrate that ReCo-WAM achieves [performance improvements], with notable gains in [task success rate and execution efficiency] on long-horizon manipulation tasks.