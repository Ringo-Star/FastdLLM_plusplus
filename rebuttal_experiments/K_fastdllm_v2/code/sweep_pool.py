#!/usr/bin/env python3
"""Shared-box-aware sweep dispatcher for Fast-dLLM-v2 + Fréchet.

The p5 is SHARED with the user's other experiments. This dispatcher only ever
launches on GPUs that are currently FREE (memory.used < --free-mib) AND that it is
not already using, so it never co-locates on a GPU the user (or another of our runs)
is occupying. A freed GPU is picked up on the next poll; if the user grabs GPUs,
we simply use fewer. No GPU is ever forcibly claimed.

Usage:
  sweep_pool.py --tasks gsm8k mbpp --limit 500 \
     --model-path $ROOT/models/Fast_dLLM_v2_7B --model-name Fast_dLLM_v2_7B \
     --outroot ~/fastdllmpp_rebuttal/phaseB --maxnew 256 [--subs 4 8 16 32] \
     [--deltas 0.00 0.10 0.20 0.25 0.30 0.50] [--free-mib 4000] [--reserve 0]
"""
import argparse, os, subprocess, sys, time, shlex

RUN = os.path.expanduser("~/fastdllmpp_rebuttal/v2_frechet/run_v2_single.sh")


def free_gpus(threshold_mib):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=30)
    except Exception:
        return []
    free = []
    for line in out.strip().splitlines():
        idx, used = [x.strip() for x in line.split(",")]
        if int(used) < threshold_mib:
            free.append(int(idx))
    return free


def build_jobs(tasks, subs, deltas):
    jobs = []
    for t in tasks:
        nf = 3 if t.startswith("mbpp") else (4 if t.startswith("minerva") else 0)
        for s in subs:
            jobs.append((t, s, "default", "-1", "-1", "0.9", nf))
            jobs.append((t, s, "threshold", "-1", "-1", "0.9", nf))
            jobs.append((t, s, "factor", "-1", "1.0", "0.9", nf))
            for d in deltas:
                jobs.append((t, s, "frechet", d, "-1", "0.9", nf))
                jobs.append((t, s, "matched_factor", d, "-1", "0.9", nf))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--limit", default="500")
    ap.add_argument("--model-path", default="$ROOT/models/Fast_dLLM_v2_1.5B")
    ap.add_argument("--model-name", default="Fast_dLLM_v2_1.5B")
    ap.add_argument("--outroot", default=os.path.expanduser("~/fastdllmpp_rebuttal/phaseX"))
    ap.add_argument("--maxnew", default="256")
    ap.add_argument("--subs", nargs="+", type=int, default=[4, 8, 16, 32])
    ap.add_argument("--deltas", nargs="+", default=["0.00", "0.10", "0.20", "0.25", "0.30", "0.40", "0.50"])
    ap.add_argument("--free-mib", type=int, default=4000)
    ap.add_argument("--reserve", type=int, default=0, help="always leave this many free GPUs for the user")
    ap.add_argument("--poll", type=int, default=15)
    args = ap.parse_args()

    outroot = os.path.expanduser(args.outroot)
    os.makedirs(os.path.join(outroot, "logs"), exist_ok=True)
    jobs = build_jobs(args.tasks, args.subs, args.deltas)
    print(f"[pool] {len(jobs)} jobs tasks={args.tasks} model={args.model_name} limit={args.limit} "
          f"free_mib<{args.free_mib} reserve={args.reserve}  {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
          flush=True)

    env_base = dict(os.environ, MODEL_PATH=args.model_path, MODEL_NAME=args.model_name, MAXNEW=args.maxnew)
    running = {}  # gpu -> (proc, tag)
    qi = 0
    while qi < len(jobs) or running:
        # reap
        for g in list(running):
            proc, tag = running[g]
            if proc.poll() is not None:
                print(f"[pool] done gpu{g} rc={proc.returncode} {tag}", flush=True)
                del running[g]
        # dispatch to free GPUs we are not already using
        avail = [g for g in free_gpus(args.free_mib) if g not in running]
        if args.reserve > 0 and len(avail) > args.reserve:
            avail = avail[: max(0, len(avail) - args.reserve)]
        while qi < len(jobs) and avail:
            g = avail.pop(0)
            t, s, sel, d, f, th, nf = jobs[qi]
            tag = f"{t}_s{s}_{sel}_d{d}_f{f}"
            logf = os.path.join(outroot, "logs", f"{tag}.log")
            cmd = ["bash", RUN, str(g), t, str(s), sel, str(d), str(f), str(th), str(nf), str(args.limit), outroot]
            with open(logf, "w") as lf:
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, env=env_base, start_new_session=True)
            running[g] = (proc, tag)
            qi += 1
            print(f"[pool] launch gpu{g} ({qi}/{len(jobs)}) {tag}", flush=True)
        time.sleep(args.poll)

    print(f"[pool] ALL DONE {args.tasks}  {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
    open(os.path.join(outroot, f"POOL_{'_'.join(args.tasks)}_COMPLETE"), "w").close()


if __name__ == "__main__":
    main()
