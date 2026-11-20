"""CPU tests and shell syntax only; never launch the actual workflow."""
import datetime
import subprocess
import unittest
from stage_III.robotwin.common import WORK,atomic_json

def main():
    for path in sorted((WORK/"scripts").glob("*.sh")):
        subprocess.run(["bash","-n",str(path)],check=True)
    suite=unittest.defaultTestLoader.discover(str(WORK/"tests"),pattern="test_*.py")
    with (WORK/"reports/cpu_tests.log").open("w") as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    report=dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                status="passed" if result.wasSuccessful() else "failed",
                tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                skipped=len(result.skipped),shell_syntax="passed",
                training_started=False,gpu_model_loaded=False,simulator_started=False,
                scope="CPU synthetic tensors and actual wrapper control flow with small stand-ins")
    atomic_json(WORK/"reports/cpu_tests.json",report)
    print((WORK/"reports/cpu_tests.log").read_text(),end="")
    print(report)
    if not result.wasSuccessful():raise SystemExit(1)
if __name__=="__main__":main()
