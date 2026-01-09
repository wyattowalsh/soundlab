# SoundLab Build Task List

**Optimization Target:** Massively parallel OpenAI GPT Codex sessions  
**Batch Strategy:** Maximize parallelism within each iteration; minimize cross-batch dependencies  
**Task Granularity:** Single file or tightly-coupled file pair per task  

---

## Execution Protocol

```
For each batch:
  1. Split tasks into parallel lanes (each lane can run independently)
  2. Spawn N Codex sessions (one per lane)
  3. Each session receives: lane task list, PRD sections, dependency outputs
  4. Sessions edit only their lane files and report changes (no commits unless requested)
  5. Barrier sync: wait for all lanes to complete
  6. Integrate changes into main working tree
  7. Proceed to next batch
```

---

## Parallelization Rules (Codex Lanes)

- One Codex session per lane; tasks in a lane are done sequentially by that session.
- Do not edit files owned by another lane in the same batch.
- If a change requires touching a shared file (for example `__init__.py`), keep all edits to that
  file in a single lane to avoid conflicts.
- Keep tightly coupled cross-file edits within the same lane.
- Use the barrier sync to reconcile conflicts before starting the next batch.

---

## Lane Status Legend

- **Unclaimed:** Lane line ends after the path.
- **CLAIMED:** Append ` — CLAIMED by <session_name> at <YYYY-MM-DD HH:MM>` to the lane line.
- **DONE:** Replace CLAIMED with `DONE by <session_name> at <YYYY-MM-DD HH:MM>`.

Only edit the single lane line you are claiming or completing.

Example:
- `- Batch 3 Lane 3C (Analysis models and feature extractors): `tasklists/lanes/batch-03-lane-3C.md` — CLAIMED by codex-07 at 2026-01-09 10:42`

---

## Session Start Checklist (Required)

1. Open `soundlab-tasklist.md`.
2. Find the earliest batch with any lane not marked DONE.
3. Pick the first unclaimed lane in that batch.
4. Claim it immediately by appending the CLAIMED tag to that lane line.
5. Open the lane file and execute tasks sequentially.

If you cannot claim a lane because all lanes are CLAIMED or DONE, stop and report status.

---

## Session Completion Checklist (Required)

1. Ensure only lane-owned files were edited.
2. Summarize changes with file paths and brief notes.
3. Update the lane line to DONE with timestamp in `soundlab-tasklist.md`.

---

## Source of Truth

- Lane ownership and status live **only** in `soundlab-tasklist.md`.
- Do not create extra claim files or lane status files.
- Lane task details live in `tasklists/lanes/`.

---

## Batch 0: Repository Skeleton (12 tasks, fully parallel)

All tasks in this batch have **zero dependencies** and create foundational files.


### Lane files
- Batch 0 Lane 0A (Core configuration): `tasklists/lanes/batch-00-lane-0A.md` — DONE by codex-01 at 2026-01-08 23:52
- Batch 0 Lane 0B (Project documentation): `tasklists/lanes/batch-00-lane-0B.md` — DONE by codex-21 at 2026-01-08 23:56
- Batch 0 Lane 0C (Repo structure and hygiene): `tasklists/lanes/batch-00-lane-0C.md` — DONE by codex-01 at 2026-01-08 23:55
**Barrier: All B0 tasks must complete before B1**

## Batch 1: Core Module (8 tasks, fully parallel)

Dependencies: B0 complete (directory structure exists)


### Lane files
- Batch 1 Lane 1A (Core models and config): `tasklists/lanes/batch-01-lane-1A.md` — DONE by codex-01 at 2026-01-09 00:02
- Batch 1 Lane 1B (Utils (GPU, logging, retry)): `tasklists/lanes/batch-01-lane-1B.md` — DONE by codex-21 at 2026-01-09 00:02
**Barrier: All B1 tasks must complete before B2**

## Batch 2: Utility Completion + I/O Module (6 tasks, fully parallel)

Dependencies: B1 complete (core types, exceptions available)


### Lane files
- Batch 2 Lane 2A (Utils progress and exports): `tasklists/lanes/batch-02-lane-2A.md` — DONE by codex-01 at 2026-01-09 00:07
- Batch 2 Lane 2B (Audio I/O): `tasklists/lanes/batch-02-lane-2B.md` — DONE by codex-21 at 2026-01-09 00:10
- Batch 2 Lane 2C (MIDI I/O and export): `tasklists/lanes/batch-02-lane-2C.md` — DONE by codex-01 at 2026-01-09 00:20
**Barrier: All B2 tasks must complete before B3**

## Batch 3: Feature Modules - Models Only (11 tasks, fully parallel)

Dependencies: B2 complete (core + utils + io available)  
Strategy: Create all Pydantic models first, then implementations


### Lane files
- Batch 3 Lane 3A (Separation + transcription scaffolding): `tasklists/lanes/batch-03-lane-3A.md` — DONE by codex-21 at 2026-01-09 00:41
- Batch 3 Lane 3B (Effects models): `tasklists/lanes/batch-03-lane-3B.md` — DONE by codex-01 at 2026-01-09 00:38
- Batch 3 Lane 3C (Analysis models and feature extractors): `tasklists/lanes/batch-03-lane-3C.md` — DONE by codex-01 at 2026-01-09 00:41
- Batch 3 Lane 3D (Voice + pipeline models): `tasklists/lanes/batch-03-lane-3D.md` — DONE by codex-01 at 2026-01-09 01:00
**Barrier: All B3 tasks must complete before B4**

## Batch 4: Feature Modules - Implementations (16 tasks, fully parallel)

Dependencies: B3 complete (all models available)


### Lane files
- Batch 4 Lane 4A (Separation implementations): `tasklists/lanes/batch-04-lane-4A.md` — DONE by codex-21 at 2026-01-09 01:19
- Batch 4 Lane 4B (Transcription implementations): `tasklists/lanes/batch-04-lane-4B.md` — DONE by codex-01 at 2026-01-09 01:05
- Batch 4 Lane 4C (Analysis implementations): `tasklists/lanes/batch-04-lane-4C.md` — DONE by codex-01 at 2026-01-09 01:15
- Batch 4 Lane 4D (Effects implementations): `tasklists/lanes/batch-04-lane-4D.md` — DONE by codex-01 at 2026-01-09 01:22
- Batch 4 Lane 4E (Pipeline implementations): `tasklists/lanes/batch-04-lane-4E.md` — DONE by codex-01 at 2026-01-09 01:28
**Barrier: All B4 tasks must complete before B5**

## Batch 5: Voice Module + Package Root (6 tasks, fully parallel)

Dependencies: B4 complete (all feature modules available)


### Lane files
- Batch 5 Lane 5A (Voice module): `tasklists/lanes/batch-05-lane-5A.md` — DONE by codex-21 at 2026-01-09 01:43
- Batch 5 Lane 5B (Package root): `tasklists/lanes/batch-05-lane-5B.md` — DONE by codex-01 at 2026-01-09 01:38
- Batch 5 Lane 5C (CLI): `tasklists/lanes/batch-05-lane-5C.md` — DONE by codex-01 at 2026-01-09 01:45
**Barrier: All B5 tasks must complete before B6**

## Batch 6: Unit Tests (15 tasks, fully parallel)

Dependencies: B5 complete (package fully implemented)


### Lane files
- Batch 6 Lane 6A (Fixtures and shared test data): `tasklists/lanes/batch-06-lane-6A.md` — DONE by codex-01 at 2026-01-09 01:48
- Batch 6 Lane 6B (Core + utils tests): `tasklists/lanes/batch-06-lane-6B.md` — DONE by codex-01 at 2026-01-09 02:46
- Batch 6 Lane 6C (IO + separation + transcription tests): `tasklists/lanes/batch-06-lane-6C.md` — DONE by codex-01 at 2026-01-09 02:42
- Batch 6 Lane 6D (Effects + analysis tests): `tasklists/lanes/batch-06-lane-6D.md` — DONE by codex-01 at 2026-01-09 02:37
- Batch 6 Lane 6E (Pipeline tests): `tasklists/lanes/batch-06-lane-6E.md` — DONE by codex-01 at 2026-01-09 02:37
**Barrier: All B6 tasks must complete before B7**

## Batch 7: Integration Tests (5 tasks, fully parallel)

Dependencies: B6 complete (unit tests validate components)


### Lane files
- Batch 7 Lane 7A (Separation + transcription integration): `tasklists/lanes/batch-07-lane-7A.md` — DONE by codex-01 at 2026-01-09 02:50
- Batch 7 Lane 7B (Analysis integration): `tasklists/lanes/batch-07-lane-7B.md` — DONE by codex-01 at 2026-01-09 02:53
- Batch 7 Lane 7C (Pipeline integration and QA selection): `tasklists/lanes/batch-07-lane-7C.md` — DONE by codex-01 at 2026-01-09 02:54
**Barrier: All B7 tasks must complete before B8**

## Batch 8: Notebook Implementation (13 tasks, fully parallel)

Dependencies: B7 complete (package tested and working)


### Lane files
- Batch 8 Lane 8A (Notebook scaffolding and setup cells (avoid overlapping cell ranges)): `tasklists/lanes/batch-08-lane-8A.md` — DONE by codex-01 at 2026-01-09 02:59
- Batch 8 Lane 8B (Input ingestion + canonical decode): `tasklists/lanes/batch-08-lane-8B.md` — DONE by codex-01 at 2026-01-09 02:59
- Batch 8 Lane 8C (Separation + candidate selection): `tasklists/lanes/batch-08-lane-8C.md` — DONE by codex-01 at 2026-01-09 03:01
- Batch 8 Lane 8D (Post-processing + transcription + MIDI cleanup): `tasklists/lanes/batch-08-lane-8D.md` — DONE by codex-01 at 2026-01-09 03:07
- Batch 8 Lane 8E (QA, preview, and export): `tasklists/lanes/batch-08-lane-8E.md` — DONE by codex-21 at 2026-01-09 03:08
**Barrier: All B8 tasks must complete before B9**

## Batch 9: CI/CD + Documentation (8 tasks, fully parallel)

Dependencies: B8 complete (notebook implemented)


### Lane files
- Batch 9 Lane 9A (CI workflows): `tasklists/lanes/batch-09-lane-9A.md` — DONE by codex-01 at 2026-01-09 03:10
- Batch 9 Lane 9B (Tooling and docs config): `tasklists/lanes/batch-09-lane-9B.md` — DONE by codex-01 at 2026-01-09 03:10
- Batch 9 Lane 9C (Guides): `tasklists/lanes/batch-09-lane-9C.md` — DONE by codex-01 at 2026-01-09 03:12
**Barrier: All B9 tasks must complete before B10**

## Batch 10: Polish + Examples (6 tasks, fully parallel)

Dependencies: B9 complete (CI + docs in place)


### Lane files
- Batch 10 Lane 10A (Example notebooks): `tasklists/lanes/batch-10-lane-10A.md` — DONE by codex-01 at 2026-01-09 03:16
- Batch 10 Lane 10B (Scripts): `tasklists/lanes/batch-10-lane-10B.md` — DONE by codex-01 at 2026-01-09 03:20
- Batch 10 Lane 10C (README polish): `tasklists/lanes/batch-10-lane-10C.md` — DONE by codex-01 at 2026-01-09 03:19

## Batch 11: Final Validation (3 tasks, sequential)

Dependencies: B10 complete  
**Note:** These run sequentially for final validation


### Lane files
- Batch 11 Lane 11A (Sequential validation (single lane)): `tasklists/lanes/batch-11-lane-11A.md` — DONE by codex-01 at 2026-01-09 03:25
