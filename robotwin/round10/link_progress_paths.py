from pathlib import Path
import time
root=Path("/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/robotwin/round10/online/jobs")
while True:
    for job in root.iterdir():
        if not job.is_dir(): continue
        direct=job/"tasks"
        if direct.exists() or direct.is_symlink(): continue
        candidates=[x for x in job.rglob("tasks") if x.is_dir()]
        if candidates:
            try: direct.symlink_to(candidates[0], target_is_directory=True)
            except FileExistsError: pass
    time.sleep(5)
