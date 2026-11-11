"""Campaign runner inside one allocation: runs the tasks of a study as srun steps in slots of a quarter node.

    python -m xcsb.campaign --study m1_ladder --run RUN --deadline EPOCH [--slots 4]

A study module provides tasks(run) (ordered dicts with id, width in quarter nodes, est_s), done(run, task),
failed(run, task) and command(run, task) (the xcsb CLI arguments). Scheduling: the first pending task (in study order) whose
estimate fits the remaining time is started when its width is free (head of line: later tasks do not overtake it for slots);
tasks that do not fit the remaining time are skipped in this allocation. Every step ends at the deadline (timeout), so the
allocation ends on its own. The runner exits when nothing runs and nothing more can start.

Claims make several allocations safe on one run directory: a task is claimed by mkdir RUN/claims/<id> (atomic) with the job
id inside. A claim is released when its step ended without a result (killed at the deadline); a step that ended with a
nonzero exit code and no record is marked failed (never retried). At start, claims of jobs that no longer exist (squeue)
and that hold no result are removed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time

SAFETY = 1.25            # a task starts only if SAFETY * est_s fits the remaining time
POLL_S = 20.0
RANKS_PER_SLOT = 8       # a quarter node: 8 ranks x 8 logical CPUs (4 cores), OMP_NUM_THREADS = 4
SRUN = ["srun", "-N", "1", "-c", "8", "--exact", "--cpu-bind=cores"]


def alive_jobs():
    out = subprocess.check_output(["squeue", "-h", "-u", os.environ["USER"], "-o", "%A"], text=True)
    return {x.strip() for x in out.split() if x.strip()}


class Runner:
    def __init__(self, study, run, deadline, slots=4, jobid=None, launch=None, alive=alive_jobs, poll=POLL_S):
        self.study, self.run, self.deadline, self.slots = study, run, float(deadline), int(slots)
        self.jobid = str(jobid or os.environ.get("SLURM_JOB_ID", "nojob"))
        self.launch = launch or self._srun
        self.alive, self.poll = alive, poll
        self.claims = os.path.join(run, "claims")
        self.logs = os.path.join(run, "logs")
        os.makedirs(self.claims, exist_ok=True)
        os.makedirs(self.logs, exist_ok=True)
        self.events = os.path.join(run, f"campaign_{self.jobid}.jsonl")

    # -- bookkeeping ---------------------------------------------------------------------------------------------
    def event(self, **kw):
        with open(self.events, "a") as fh:
            fh.write(json.dumps(dict(t=time.time(), job=self.jobid, **kw)) + "\n")

    def _claim_dir(self, task):
        return os.path.join(self.claims, task["id"])

    def claim(self, task):
        try:
            os.mkdir(self._claim_dir(task))
        except FileExistsError:
            return False
        open(os.path.join(self._claim_dir(task), "job"), "w").write(self.jobid)
        return True

    def claimed(self, task):
        return os.path.isdir(self._claim_dir(task))

    def marked_failed(self, task):
        return os.path.exists(os.path.join(self._claim_dir(task), "failed"))

    def clean_stale(self, tasks):
        alive = self.alive()
        for t in tasks:
            d = self._claim_dir(t)
            if not os.path.isdir(d) or self.marked_failed(t) or self.study.done(self.run, t):
                continue
            job = open(os.path.join(d, "job")).read().strip() if os.path.exists(os.path.join(d, "job")) else ""
            if job != self.jobid and job not in alive:
                shutil.rmtree(d)
                self.event(kind="stale_claim_removed", task=t["id"], claim_job=job)

    # -- steps ---------------------------------------------------------------------------------------------------
    def _srun(self, task, seconds, log):
        cmd = (["timeout", "--signal=TERM", "--kill-after=60", str(int(seconds))] + SRUN
               + ["-n", str(RANKS_PER_SLOT * task["width"]), sys.executable, "-m", "xcsb.cli"] + self.study.command(self.run, task))
        return subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT)

    def loop(self):
        tasks = self.study.tasks(self.run)
        self.clean_stale(tasks)
        running, free = {}, self.slots
        self.event(kind="start", deadline=self.deadline, slots=self.slots, n_tasks=len(tasks))
        while True:
            for tid, (p, task, t0) in list(running.items()):
                rc = p.poll()
                if rc is None:
                    continue
                free += task["width"]
                del running[tid]
                ok, bad = self.study.done(self.run, task), self.study.failed(self.run, task)
                if not ok and not bad and rc not in (124, 137, -15, -9, 143):
                    open(os.path.join(self._claim_dir(task), "failed"), "w").write(f"exit code {rc}, no record\n")
                    bad = True
                if not ok and not bad:
                    shutil.rmtree(self._claim_dir(task))       # ended at the deadline without a result: free for later
                self.event(kind="end", task=tid, rc=rc, ok=ok, failed=bad, seconds=time.time() - t0)
            remaining = self.deadline - time.time()
            tasks = self.study.tasks(self.run)                 # a study may add tasks once earlier ones are done (stages)
            pending = [t for t in tasks if not self.study.done(self.run, t) and not self.study.failed(self.run, t)
                       and not self.claimed(t)]
            started = False
            for t in pending:
                if SAFETY * t["est_s"] > remaining:
                    continue
                if t["width"] > free:
                    break
                if not self.claim(t):
                    continue
                log = os.path.join(self.logs, f"{t['id']}.log")
                running[t["id"]] = (self.launch(t, remaining, log), t, time.time())
                free -= t["width"]
                started = True
                self.event(kind="launch", task=t["id"], width=t["width"], est_s=t["est_s"], remaining_s=remaining)
            if not running and not started:
                fits = [t["id"] for t in pending if SAFETY * t["est_s"] <= remaining]
                self.event(kind="exit", pending=len(pending), pending_fitting=len(fits), remaining_s=remaining)
                return
            time.sleep(self.poll)


def main(argv=None):
    p = argparse.ArgumentParser(prog="xcsb.campaign")
    p.add_argument("--study", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--deadline", type=float, required=True, help="epoch seconds at which every step is stopped")
    p.add_argument("--slots", type=int, default=4)
    a = p.parse_args(argv)
    study = importlib.import_module(f"xcsb.studies.{a.study}")
    Runner(study, a.run, a.deadline, a.slots).loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
