"""Use the retained rollout core without changing shared ablation modules."""
from pathlib import Path
import numpy as np

# LIBERO-Plus fog uses the historical alias, identical to float64.
# Keep this compatibility change local to A2 simulator processes.
if "float_" not in np.__dict__:
    np.float_=np.float64

from ablation.aba1.plus_common import environment

environment.BASE=Path(__file__).resolve().parents[5]

if __name__=="__main__":
    from ablation.aba1.plus_common.rollout import main
    main()
