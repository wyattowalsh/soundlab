# SoundLab canonicalization plan

SoundLab remains the canonical public repository and brand.

## Current state

- `notebooks/soundlab_studio.ipynb` is the current interactive Colab entrypoint.
- `soundlab.ipynb` is an older root entrypoint.
- `packages/soundlab/src/soundlab/` is the reusable typed engine.
- PyPI's `soundlab` distribution belongs to another project; this repository must not direct users to install or publish that distribution.
- The repository does not currently contain a reviewed `uv.lock`.

## Target state

- One root `soundlab.ipynb` as the canonical generated product artifact.
- Package source owns reusable logic.
- A deterministic notebook builder detects drift.
- Safe first run: synthetic sample, no Drive mount, no public app, no hosted upload, no paid processing.
- Shared song/project state across separation, sections, MIDI, description, conversion, and export.
- Current notebook and package behavior covered by unit, integration, notebook, browser/component, and packaging checks.

## Migration stages

1. **Safety and truth**
   - remove incorrect PyPI install and publishing paths;
   - make persistence and external effects explicit;
   - repair workflow evidence labels;
   - replace historical session logs with durable policy.
2. **Canonical notebook**
   - migrate the latest reviewed workbench into the package;
   - add deterministic notebook generation;
   - retain both existing notebooks until parity is verified.
3. **Shared state and validation**
   - typed project, asset, operation, job, and result state;
   - disabled features impose no validation;
   - readiness means executable.
4. **Song workspace UX**
   - persistent waveform and transport;
   - contextual actions;
   - synchronized A/B audition;
   - clear draft/applied/running/stale/result states;
   - project save/open with source relinking and hash verification.
5. **Release candidate**
   - clean dependency resolution and committed lock;
   - fresh managed-Colab install;
   - representative GPU/model runs;
   - perceptual review;
   - exact-byte release packaging and GitHub Release.

## Evidence policy

CPU/static CI does not establish managed-Colab rendering, GPU fit, checkpoint access, transcription accuracy, source-separation quality, or hosted-service behavior. Each release must report these independently.
