# SoundLab Task Lane: Batch 9 Lane 9A - CI workflows

**Batch:** 9
**Lane:** 9A
**Title:** CI workflows
**Dependencies:** B8 complete (notebook implemented)
**Barrier:** All B9 tasks must complete before B10

## Tasks
- `B9.01` Create CI workflow. Output: `.github/workflows/ci.yml`. Spec: lint, typecheck, test jobs per PRD §8.1
- `B9.02` Create release workflow. Output: `.github/workflows/release.yml`. Spec: Build + PyPI publish on tag per PRD §8.2
- `B9.03` Create Colab test workflow. Output: `.github/workflows/colab-test.yml`. Spec: Weekly scheduled test of notebook imports

## Lane Execution Rules
- Claim this lane in `soundlab-tasklist.md` before editing files.
- Edit only files listed in this lane.
- Do not edit other batches or lane files.
- When done, mark the lane DONE in `soundlab-tasklist.md`.
