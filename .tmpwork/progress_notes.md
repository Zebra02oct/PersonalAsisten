# Progress notes — biometric matching fix

Repo: D:\project\python\asisten (no git commits)

## Root cause (confirmed)
`src/face_recognition/feature_extractor.py::_normalize_features` does a per-sample min-max
over all 30 features -> every signature has min=0/max=1 and all components non-negative
-> all signatures live in a near-parallel cone -> cosine similarity useless (0.88..0.91 for
random geometry vs 0.996..0.999 for same person). Login threshold 0.60 => accepts anyone.

## Plan / status
- [ ] Feature extractor: scale-invariant (divide raw-length dims by per-frame face-size ref),
      remove min-max. 30 features confirmed: dist(12) ratio(4) angle(4) sym(8) prop(2).
- [ ] New shared module `src/face_recognition/matcher.py`: z-score (per-dimension std from
      enrollment) RMS distance -> confidence; ONE threshold constant.
- [ ] Signature storage v2: {"features":[30], "std":[30], "version":2}
- [ ] Legacy detection + re-enrollment signal (DB + JSON enrollment manager)
- [ ] Wire face_login.py, recognizer.py, enrollment.py to the shared matcher
- [ ] get_feature_count() 32 -> 30
- [ ] connection.py: @contextmanager on get_session
- [ ] create_tables/create_database on startup (test with scratch DB)
- [ ] face_login Stage 3: fetch signature once (no per-frame DB hit)
- [ ] login_system/enrollment.py: real per-pose validation, real captured frame for eye shape,
      don't close parent-owned detector
- [ ] KEY_LANDMARKS left_eye_top/left_eye_center duplicate -> verify empirically
- [ ] eye_shape_classifier: corner-index reuse + degenerate iris visibility
- [ ] tests/ (pytest)
- [ ] README targeted edits

## Scratch space
`.tmpwork/` — text artifacts only. NEVER open/write images.
