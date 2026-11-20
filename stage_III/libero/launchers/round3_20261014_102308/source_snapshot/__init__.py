"""Isolated Stage-III classic LIBERO GRPO training package.

This package never modifies the existing Stage-I/II implementation. It loads
the completed Stage-II Student checkpoint read-only and adds only the
reward-trained GateActor used by Stage III.
"""
