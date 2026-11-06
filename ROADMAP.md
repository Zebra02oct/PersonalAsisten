# Personal Assistant Dashboard - Roadmap

## Current status

**The project is called "Personal Assistant Dashboard", but only the
biometric-login half of it exists. The dashboard itself is unbuilt.**

That mismatch is the single most important fact about this repository. There
is no dashboard, no weather panel, no GitHub panel, no REST API and no mobile
client. What *is* built and working is a face-recognition login pipeline with
a sci-fi eye scanner, an enrollment flow and MySQL persistence.

Read the phase list below as "how far along the biometric half is", not as
"how far along the product is". Phases 2-5 are the actual product and none of
them have been started.

---

## Phase 0 - Project Setup

**Status: DONE**

- [x] Virtual environment + `requirements.txt`
- [x] `.gitignore` (biometric data excluded)
- [x] `.env.example` template (secret-free)
- [x] Project structure
- [x] MySQL database `asisten_db` reachable
- [ ] **First git commit** (repo still has zero commits on `master`)

## Phase 1 - Face Detection

**Status: DONE** (`src/face_detection/detector.py`)

- [x] Webcam capture + preview
- [x] Haar cascade detection with bounding boxes
- [x] Face count overlay
- [x] Clean exit (`q` / ESC) with camera release
- [x] Graceful failure when no camera is available
- [x] Cascade path resolved from `__file__` (independent of the working directory)

Known limits: lighting and camera quality affect detection; side profiles and
occlusion reduce the hit rate.

## Phase 2 - Dashboard Foundation

**Status: NOT STARTED**

- [ ] Service module structure
- [ ] Configuration loader
- [ ] Basic dashboard layout (Tkinter or web)
- [ ] Module registration system

## Phase 3 - API Integrations

**Status: NOT STARTED**

- [ ] Weather API integration
- [ ] GitHub API integration
- [ ] Task management module
- [ ] Schedule / reminder system
- [ ] Password generator
- [ ] Secure API key handling (placeholder vars already in `.env.example`)

## Phase 4 - Web Interface

**Status: NOT STARTED**

- [ ] Backend API (Flask or FastAPI)
- [ ] Browser dashboard
- [ ] Service endpoints
- [ ] Responsive design

## Phase 5 - Flutter Mobile Client

**Status: NOT STARTED**

- [ ] Flutter project setup
- [ ] API contract documentation
- [ ] Mobile dashboard
- [ ] Camera features

Note: a phone's `localhost` is not the laptop's `localhost` - the mobile client
must call the laptop's LAN IP. Decide at this phase whether processing runs on
the laptop or on the device.

## Phase 6+ - Biometric & Recognition

**Status: MOSTLY DONE** - this is the part that actually exists.

### Landmarks & scanning

- [x] MediaPipe FaceMesh, 468 landmarks (478 with `refine_landmarks=True`)
- [x] Sci-fi eye biometric scanner overlay (`src/face_analysis/biometric_scanner.py`)
- [x] Biometric readout derived from the tracked eye geometry (deterministic,
      stable frame to frame; mm values are relative - a webcam frame carries
      no absolute scale reference)

### Enrollment & matching

- [x] Geometric face signature, 30 features
- [x] Guided enrollment, 5 samples
- [x] Staged 3-stage verification
- [x] MySQL storage
- [x] MySQL verified working: 1 enrolled owner profile + 6 login-history rows

### Not done in this phase

- [ ] Head-pose estimation
- [ ] Real head-pose-based pose validation (currently faked - see defects)
- [ ] Multi-user support (the schema is single-owner **by design**)
- [ ] Tests for the matching math - **IN PROGRESS**

---

## Known open defects

| # | Defect | Status |
|---|--------|--------|
| 1 | Enrollment validates pose with a hard-coded/stubbed check instead of real head-pose data | OPEN |
| 2 | Face-signature feature count mismatch: the signature is 30 features, the consumer expects 32 | OPEN |
| 3 | `create_tables()` is never invoked - tables only exist because they were created by hand | IN PROGRESS |
| 4 | Missing `@contextmanager` on the DB session/transaction helper | IN PROGRESS |
| 5 | Stage 3 verification runs a DB query on every frame | IN PROGRESS |
| 6 | Duplicate OpenCV pins (`opencv-python` + `opencv-contrib-python`) | **DONE** - `requirements.txt` now pins `opencv-contrib-python` only |
| 7 | Five orphan DB tables with no ORM model: `attendance_logs`, `face_profiles`, `recognition_events`, `system_settings`, `users` (all empty) | OPEN |
| 8 | Repository has zero commits; every file is untracked | OPEN |

---

## Notes

- Verify each phase before starting the next one.
- Biometric data (`src/face_recognition/data/*.json`) must never be committed -
  it contains enrolled face signatures.
- The `.env` file holds real credentials and is git-ignored; `.env.example`
  must stay secret-free.
- No premature optimisation or over-engineering.
