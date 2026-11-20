"""Reuse the audited student server with the A1 numerical backend settings."""
from a1_local.numerics import configure_attention_backends
from ablation.aba1.plus_common.server import main

if __name__=="__main__":
    configure_attention_backends()
    main()
