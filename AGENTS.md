# Development contract

- Work only in this independent data pipeline repository. Read README.md first.
- Raw sources, frozen selections, v1.2 assets, and running experiments are immutable.
- Preserve source revision, hashes, timing, frame/quaternion conventions, source groups and splits.
- Geometric expected contact is not actual pair force ground truth. Unknown is never silently negative.
- No automatic re-cooking, mesh recentering, joint-order guessing, rescaling or clip cutting.
- Keep schema, source labels, native geometry binding and generated labels separately versioned.
- Run CPU numerical, adapter, and geometry tests plus real-source smoke before batch submission.
- Use a new frozen source snapshot and output root for each production batch. Read queue and claims first.
- Code commits are authorized. Do not commit data, results, assets or credentials; no automatic push.
- Slurm offline labeling is authorized. Training, experiment cancellation and system/runtime modifications outside this pipeline are not.
