# SoundLab Task Lane: Batch 7 Lane 7C - Pipeline integration and QA selection

**Batch:** 7
**Lane:** 7C
**Title:** Pipeline integration and QA selection
**Dependencies:** B6 complete (unit tests validate components)
**Barrier:** All B7 tasks must complete before B8

## Tasks
- `B7.04` Test pipeline integration. Output: `tests/integration/test_pipeline.py`. Spec: Full pipeline: upload -> separate -> analyze -> effects -> export; verify zip contents
- `B7.05` Test pipeline QA selection. Output: `tests/integration/test_pipeline_qa_selection.py`. Spec: Excerpt candidates scored; best candidate chosen; resume skips completed stages

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
