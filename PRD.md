# Gesture Media Controller: Product Requirements Document

Author: Nirav Mehta | Status: Draft v1 | Audience: autonomous coding agent (long-running, able to run tests)

## 1. Overview

A Python desktop tool that uses the laptop webcam to recognize hand gestures and control media playback on Linux. Everything runs locally on CPU. It runs as a terminal app with an OpenCV preview window.

Controls: play/pause, volume, mute, next, previous, seek, pitch, open player.

## 2. Goals and Non-Goals

### Goals

- Reliable, low-latency gesture control on a CPU-only laptop
- Linux (Wayland + PipeWire) first, with an architecture ready for Windows and macOS backends
- Composable player layer: mpv built-in player and MPRIS players (VLC, Spotify, browsers) behind one interface
- Clean seams so the author can write or replace CV logic by hand
- Fully testable by an agent without a human in the loop

### Non-goals (v1)

- Web UI, mobile, cloud inference
- Custom gesture training
- Windows and macOS backends (interfaces and stubs only)
- Multi-hand or multi-user control
- Tray icon, settings GUI, systemd service

## 3. Target Environment

- OS: Linux, Wayland session, PipeWire audio with WirePlumber
- Hardware: CPU-only laptop, built-in webcam
- Python: 3.12 managed by uv. Pin `requires-python = '>=3.10,<3.13'` because mediapipe wheels do not reliably cover 3.13
- System packages, Arch: `playerctl wireplumber mpv`. Debian/Ubuntu: `playerctl wireplumber mpv libmpv2`. mpv/ffmpeg must be built with the rubberband filter
- Wayland rule: do NOT rely on synthetic key injection (pynput, xdotool). Use MPRIS via `playerctl` and `wpctl` only
- OpenCV window: if the preview fails on Wayland, run with `QT_QPA_PLATFORM=xcb`

## 4. Dependencies

- Python (via uv): `mediapipe`, `numpy`, `python-mpv`, `PyYAML`, dev: `pytest`, `ruff`
- Do not add `opencv-python` separately. `mediapipe` already installs `opencv-contrib-python`, and installing both causes conflicts
- Model: `gesture_recognizer.task`, downloaded by `scripts/fetch_model.py` into `models/` (gitignored). Source: https://storage.googleapis.com/mediapipe-models/gesture\_recognizer/gesture\_recognizer/float16/latest/gesture\_recognizer.task
- MediaPipe built-in gesture classes: `Closed_Fist`, `Open_Palm`, `Pointing_Up`, `Thumb_Down`, `Thumb_Up`, `Victory`, `ILoveYou`, `None`

## 5. Architecture

Pipeline:

`FrameSource` -> `GestureClassifier` (MediaPipe) -> `GestureEngine` (state machine) -> `ActionDispatcher` -> `Player` and `AudioControl` backends

The preview HUD reads engine state and never mutates it.

| Module | Responsibility |
| --- | --- |
| `sources.py` | Camera, video file, and image-sequence frame sources behind one interface |
| `vision/classifier.py` | `GestureClassifier` interface plus the MediaPipe implementation. Returns an `Observation` or None |
| `vision/rules.py` | Stub for a hand-written landmark rule classifier (extension point for the author) |
| `engine.py` | Gesture state machine. Pure logic, injectable clock, no I/O |
| `actions.py` | `Action` types and dispatcher with a worker thread |
| `players/base.py` | `Player`, `AudioControl`, `Capability` |
| `players/mpris.py` | playerctl-based backend (VLC, Spotify, browsers) |
| `players/mpv.py` | Built-in mpv backend with pitch |
| `players/audio_wpctl.py` | System volume and mute via wpctl |
| `players/fake.py` | Recording fake for tests |
| `hud.py` | OpenCV overlay drawing |
| `config.py` | YAML load, defaults, validation |
| `cli.py` | Entry points |

### Threading

- Main thread: capture, inference, HUD (`cv2.imshow` must stay on the main thread)
- Worker thread: action executor fed by a queue. Subprocess calls must never block the frame loop
- Coalesce continuous actions: if several volume, seek, or pitch deltas are queued, merge them into one

## 6. Repo Layout

```
gesture-media/
  pyproject.toml
  README.md
  config.yaml
  models/                 (gitignored)
  scripts/fetch_model.py
  src/gesture_media/
    __init__.py cli.py config.py sources.py engine.py actions.py hud.py
    vision/ __init__.py classifier.py rules.py types.py
    players/ __init__.py base.py mpris.py mpv.py audio_wpctl.py fake.py
  tests/
    fixtures/             (author-recorded clip and labels)
    test_engine.py test_players_mpris.py test_players_mpv.py test_config.py
```

Entry point: `[project.scripts] gesture-media = 'gesture_media.cli:main'`. Run with `uv run gesture-media ...`.

## 7. Functional Requirements

### FR-1 Frame source

- Sources: camera index, video file, image directory
- Default 640x480; configurable width, height, target fps
- Frame is mirrored horizontally before processing so on-screen right equals the user's right
- Camera open failure: print a clear message, exit code 2
- Must release the camera on exit and on SIGINT

### FR-2 Recognition

- Use MediaPipe Tasks `GestureRecognizer` with `num_hands=1`, running mode VIDEO, `recognize_for_video(mp.Image(SRGB, rgb_frame), timestamp_ms)` with strictly increasing timestamps
- Minimum gesture score default 0.6, configurable
- Output type: `Observation(gesture: str, score: float, handedness: str, landmarks: 21 x (x, y, z) normalized, t_ms: int)`, or None when no hand is present
- Palm center for all continuous tracking = mean of landmarks 0, 5, 9, 13, 17
- Optional dominant-hand filter in config (`any`, `left`, `right`)

### FR-3 Gesture engine

See section 8. Must be deterministic given a sequence of `(Observation | None, time)` inputs.

### FR-4 Actions

`PLAY_PAUSE`, `NEXT`, `PREVIOUS`, `SEEK(delta_seconds)`, `VOLUME(delta_percent)`, `MUTE_TOGGLE`, `PITCH(set_semitones)`, `PITCH_RESET`, `OPEN_PLAYER`.

### FR-5 Players

See section 9.

### FR-6 Pitch

- Range -12 to +12 semitones, quantized to 0.5
- Pitch scale = 2 \*\* (semitones / 12)
- Implemented only in the mpv backend using the rubberband filter. Tempo must stay constant
- Startup capability check: confirm rubberband is available (for example `mpv --af=help` or the ffmpeg filter list). If missing, `doctor` reports it and pitch falls back to a flagged 'vinyl mode' (speed change, pitch shifts with tempo) shown in the HUD
- The agent must run a spike first to determine how to change pitch live with the installed mpv version (for example the `af-command` property or re-setting `af`) and pick the approach with no audible dropouts

### FR-7 Preview window and HUD

- Shows: mirrored frame, armed zone rectangle, current gesture and score, hold-progress bar, last action, fps, current volume, mute state, pitch in semitones, active backend, unsupported-capability notices
- Keys: `q` or `Esc` quit, `d` toggle detection, `h` toggle HUD, `l` toggle landmark overlay
- `--no-preview` runs headless (terminal logs only)

### FR-8 Config

YAML, validated, with defaults for everything. Unknown keys produce a warning, invalid values produce a clear error.

### FR-9 CLI

- `gesture-media run [--config PATH] [--camera N | --source FILE] [--player auto|mpv|mpris] [--media PATH ...] [--no-preview]`
- `gesture-media doctor`: checks camera access, model file, playerctl, wpctl, libmpv import, rubberband, active players, session type. Prints pass or fail per check, exit code 1 on any hard failure
- `gesture-media list-cameras`
- `gesture-media record --out run.jsonl`: logs Observations (landmarks and labels only, never images) for replay tests
- `gesture-media replay run.jsonl`: feeds a recording through the engine with a fake player and prints actions

### FR-10 Logging

Terminal log line per action, for example `12:01:03 ACTION volume +3% (system)`. `--verbose` adds per-frame debug. No frames are ever written to disk or sent over a network.

## 8. Gesture Map and Engine Spec

| Gesture (MediaPipe class) | Type | Behavior | Defaults |
| --- | --- | --- | --- |
| Open\_Palm | Discrete | Toggle play/pause | hold 500 ms, cooldown 1000 ms |
| Closed\_Fist | Discrete | Toggle mute | hold 500 ms, cooldown 1000 ms |
| Thumb\_Up | Discrete | Next track | hold 400 ms, cooldown 1000 ms |
| Thumb\_Down | Discrete | Previous track | hold 400 ms, cooldown 1000 ms |
| Pointing\_Up | Continuous | Volume: palm up = louder, down = quieter | deadzone 0.05, max 30 percent/s |
| Pointing\_Up, held still | Hold-still | Open player | 1500 ms within deadzone |
| Victory | Continuous | Seek: palm right = forward, left = back | deadzone 0.05, max 20 s/s |
| ILoveYou | Continuous | Pitch: palm up = higher, down = lower | deadzone 0.05, max 4 semitones/s |
| ILoveYou, held still | Hold-still | Reset pitch to 0 | 1500 ms within deadzone |

### Engine rules

1. Arming: the palm center must be inside the center zone (default x 0.15 to 0.85, y 0.10 to 0.90 of the frame) and detection must be enabled. Otherwise ignore the observation
2. Stability: a new gesture label must persist for `stable_ms` (default 150) before the engine switches to it. This filters flicker
3. Grace: up to `grace_ms` (default 200) of None or a different label does not count as release
4. One active gesture at a time. Switching requires passing rule 2
5. Discrete gestures: fire once after the hold time, then lock until the gesture is released, then apply the cooldown
6. Continuous gestures (joystick style, not absolute position):
   - On engage, record the anchor = palm center
   - Displacement d = palm center minus anchor along the relevant axis, smoothed with EMA (alpha 0.4)
   - Image y grows downward, so palm up = negative y change. Map so that up is positive
   - Rate = sign(d) \* clamp((abs(d) - deadzone) / (max\_disp - deadzone), 0, 1) \* max\_rate, with `max_disp` default 0.25
   - Emit delta actions every `tick_ms` (default 100)
7. Hold-still gestures: after engage, if displacement stays within the deadzone for `still_ms`, fire once and lock. Once a continuous gesture has engaged (displacement exceeded deadzone), hold-still cannot fire until re-engagement
8. Hand lost for longer than grace: reset all state, clear locks (not cooldowns)
9. All thresholds live in config. Values above are defaults

### Extension point for the author's own CV work

`GestureClassifier` is an interface. The default implementation wraps MediaPipe. `vision/rules.py` contains a stub rule-based classifier operating on the 21 landmarks that returns the same labels, so the engine does not change when the author swaps classifiers. Selected by `vision.classifier: mediapipe | rules`.

## 9. Player Layer

```python
class Capability(Enum):
    PLAY_PAUSE; NEXT; PREVIOUS; SEEK; VOLUME; MUTE; PITCH; OPEN

class Player(Protocol):
    name: str
    capabilities: frozenset[Capability]
    def is_available(self) -> bool: ...
    def play_pause(self) -> None: ...
    def next(self) -> None: ...
    def previous(self) -> None: ...
    def seek(self, seconds: float) -> None: ...
    def set_pitch(self, semitones: float) -> None: ...
    def open(self) -> None: ...
    def close(self) -> None: ...

class AudioControl(Protocol):
    def volume_delta(self, percent: float) -> None: ...
    def toggle_mute(self) -> None: ...
    def get_state(self) -> tuple[float, bool]: ...
```

### Backends

| Backend | Mechanism | Capabilities |
| --- | --- | --- |
| `MprisPlayer` | `playerctl` subprocess: `play-pause`, `next`, `previous`, `position 5+` or `5-` for seek, optional `--player NAME` | play/pause, next, prev, seek, open (runs `open_command`) |
| `MpvPlayer` | python-mpv in-process, plays `--media` files or playlist | all, including pitch and player volume |
| `WpctlAudio` | `wpctl set-volume @DEFAULT_AUDIO_SINK@ 5%+ -l 1.0`, `wpctl set-mute @DEFAULT_AUDIO_SINK@ toggle`, `wpctl get-volume` | system volume, mute |
| `FakePlayer` | records calls | all (tests) |

### Rules

- VLC is controlled through MPRIS (`MprisPlayer`). VLC pitch is not supported, so the HUD shows 'pitch unsupported on this backend'
- Volume target is configurable: `audio.volume_target: system | player`. Default: `system` for MPRIS, `player` for mpv
- `player.backend: auto` picks MPRIS if `playerctl -l` lists any player, else the built-in mpv
- Missing capability: log once, show in HUD, never crash
- All subprocess calls have a 1 second timeout and catch errors
- Composability requirement: adding a backend means implementing `Player` and registering it in a registry dict. No changes to engine or dispatcher
- Should-have: optional `routing:` map in config that sends a capability to a named backend (for example pitch to mpv, rest to mpris)
- Open player: with mpv backend, create or raise the mpv window. With MPRIS backend, run `player.open_command` (default `['vlc']`) via `subprocess.Popen`, detached
- mpv options: `input_default_bindings=False`, `input_vo_keyboard=False`, `osc=True`, `ytdl=False`, `idle=True`, `force_window=True`

## 10. Config Example

```yaml
camera:
  index: 0
  width: 640
  height: 480
  fps: 30
vision:
  classifier: mediapipe
  min_score: 0.6
  hand: any
  zone: {x_min: 0.15, x_max: 0.85, y_min: 0.10, y_max: 0.90}
engine:
  stable_ms: 150
  grace_ms: 200
  tick_ms: 100
  deadzone: 0.05
  max_disp: 0.25
  smoothing_alpha: 0.4
gestures:
  Open_Palm:   {action: play_pause, hold_ms: 500, cooldown_ms: 1000}
  Closed_Fist: {action: mute, hold_ms: 500, cooldown_ms: 1000}
  Thumb_Up:    {action: next, hold_ms: 400, cooldown_ms: 1000}
  Thumb_Down:  {action: previous, hold_ms: 400, cooldown_ms: 1000}
  Pointing_Up: {action: volume, max_rate: 30, still_action: open_player, still_ms: 1500}
  Victory:     {action: seek, max_rate: 20}
  ILoveYou:    {action: pitch, max_rate: 4, still_action: pitch_reset, still_ms: 1500}
player:
  backend: auto
  mpris_player: null
  open_command: [vlc]
  media: []
audio:
  volume_target: system
pitch:
  min_semitones: -12
  max_semitones: 12
  step: 0.5
hud:
  enabled: true
  show_landmarks: false
```

## 11. Testing and Verification

The agent must make these pass without a human:

- Engine unit tests with a fake clock and scripted Observation sequences. Cover: each discrete gesture, hold too short, flicker rejection, grace period, hand loss, cooldown, lock until release, each continuous gesture with deadzone and clamping, direction signs, hold-still firing, hold-still vs continuous exclusivity, zone arming, gesture switching
- Contract tests for `MprisPlayer` and `WpctlAudio` with subprocess mocked. Assert exact commands and timeout handling
- Config tests: defaults, overrides, invalid values, unknown keys
- mpv integration test: headless (`ao=null`, `vo=null`), play a generated sine WAV (made with ffmpeg lavfi), set pitch, assert pitch state and that tempo is unchanged. Skip automatically if libmpv or rubberband is missing
- Replay test: `gesture-media replay` on a recorded JSONL produces the expected action sequence
- Recognizer smoke test on the author's fixture clip (`tests/fixtures/gestures.mp4` plus `labels.json` with expected gestures per time range). Accept at least 85 percent frame-level agreement on stable segments
- Gates: `uv run pytest`, `uv run ruff check`, `uv run gesture-media doctor`

## 12. Non-Functional Requirements

- Frame loop: at least 20 fps at 640x480 on the target laptop. Show measured fps in HUD
- Discrete gesture to action latency under 300 ms after the hold time is met
- Average CPU under 35 percent of total on a typical 4-core laptop. Measure and report; reduce resolution or skip frames if exceeded
- Reliability: no crash when a player disappears, camera frame read fails once, or a subprocess errors
- Clean shutdown: release camera, close mpv, drain worker queue
- Privacy: frames stay in memory. `record` stores landmarks and labels only

## 13. Milestones

| Milestone | Scope | Acceptance |
| --- | --- | --- |
| M0 | uv project, deps, `fetch_model.py`, `doctor` | `doctor` runs and reports each check |
| M1 | Frame source, MediaPipe classifier, HUD | Live preview shows gesture name and score at 20+ fps |
| M2 | Engine with full gesture map and unit tests | All engine tests pass; `replay` works |
| M3 | MPRIS and wpctl backends, dispatcher with worker thread | Real gestures control a running VLC or Spotify on Wayland |
| M4 | mpv backend, pitch spike, pitch and seek via gestures | Pitch changes live without tempo change or dropouts |
| M5 | Config validation, `record`, README, demo instructions | Fresh clone to running app using README only |
| M6 (stretch) | `routing:` map, Windows and macOS stubs, tray | Not required for v1 |

## 14. Human Tasks (Nirav)

- Record `tests/fixtures/gestures.mp4` (about 60 seconds, each gesture several times with idle gaps) and write `labels.json`
- Tune thresholds for own lighting and webcam
- Optionally implement the rule-based classifier in `vision/rules.py`
- Capture the demo GIF or video for the project report

## 15. Risks and Open Questions

- Gesture confusion between Pointing\_Up and Victory or ILoveYou in poor lighting. Mitigation: stability window, score threshold, tunable config
- Some MPRIS players restart the current track on `previous`. Document this behavior
- Live pitch change in mpv may glitch depending on version. Mitigation: the M4 spike and the fallback mode
- OpenCV window under Wayland may need XWayland. Mitigation: `QT_QPA_PLATFORM=xcb`
- Webcam resolution and exposure vary. Mitigation: configurable, `list-cameras`
- Open question: should system volume or player volume be the default for the mpv backend? Default is player volume, revisit after testing

## 16. Out of Scope and Future Work

- Windows backend (pycaw, media keys), macOS backend (osascript, media keys)
- Custom gesture training and a gesture-mapping GUI
- Tray icon and systemd user service
- Multi-hand gestures, face-based controls
- Spotify Web API integration

## 17. Definition of Done

- All milestones M0 to M5 accepted
- `uv run pytest` and `uv run ruff check` pass
- `doctor` passes on the author's machine
- README covers install (system packages and uv), run, config, gesture cheat sheet, troubleshooting (Wayland, camera, rubberband)
- Demo shows all nine gesture behaviors working live with VLC or mpv

## 18. Calibration (Optional Feature)

Amends FR-9 (CLI), section 10 (config), milestone M5, and section 11 (testing). Calibration is never required. The app must work well with defaults for any user.

### Purpose

Adapt thresholds to a specific user, camera, and seating distance without training any model.

### CLI

- `gesture-media calibrate [--camera N] [--skip-gestures]`
- `gesture-media calibrate --reset` deletes the saved profile
- `gesture-media run --no-calibration` ignores the saved profile
- First run prints a one-line hint that `calibrate` exists. It never forces it

### Guided flow (preview window with on-screen prompts, about 60 to 90 seconds)

1. Rest: hold a hand in a relaxed pose for 3 seconds. Measure palm-center jitter (standard deviation). Set `deadzone = max(0.03, 3 * sigma)`
2. Range: move the palm to the comfortable top, bottom, left, and right extents, holding 1 second at each. Set `max_disp` to about 80 percent of the smaller comfortable extent, clamped to 0.15 to 0.35
3. Zone: use the measured extents plus a margin to set the arming zone, clamped inside the frame
4. Gesture check (skippable): prompt for each of the 7 gestures for 2 seconds. Record the mean score. If any gesture averages below `min_score`, warn and suggest better lighting or a lower `min_score`. Never lower it automatically below 0.5
5. Summary: print old versus new values, ask to save (`y` or `n`)

### Storage and precedence

- Saved to `~/.config/gesture-media/profile.yaml` (respect `XDG_CONFIG_HOME`)
- Precedence, lowest to highest: built-in defaults, `config.yaml`, calibration profile, CLI flags
- Profile stores only numbers (deadzone, max\_disp, zone, optional min\_score) and a version field. No images, no landmarks
- Invalid or old-version profile: warn and fall back to defaults

### Requirements

- Calibration math lives in a pure function over recorded samples so it can be unit tested without a camera
- HUD shows 'calibrated profile active' when a profile is loaded
- `doctor` reports whether a profile exists and its version

### Tests

- Unit tests with synthetic sample sets: steady hand, jittery hand, small range, large range, clamping, missing samples
- Precedence tests for config layering
- Profile load tests: corrupt file, old version, missing file

### Acceptance

- Calibration finishes in under 2 minutes, can be cancelled with `Esc` without writing anything, and an engine run with a saved profile uses the calibrated values

## 19. Run Mode, Tray Icon, and Notifications

Amends FR-7 (HUD), FR-9 (CLI), section 8 (engine rules), section 10 (config), and milestone M5.

### Run model

- Manual start only. The user launches `gesture-media run`. No auto-start and no systemd unit in v1
- Optional stretch: document a `systemd --user` unit in the README for people who want auto-start. It must not be required for anything
- `--no-preview` runs without the preview window while the tray icon stays active
- Single-instance lock (lock file in `$XDG_RUNTIME_DIR`). A second launch prints a message and exits
- Clean exit via tray Quit, Ctrl+C, or SIGTERM. Release the camera and close mpv

### Tray icon

- Must work on Wayland. The agent runs a short spike to choose between `pystray` (AppIndicator or StatusNotifierItem backend) and a Qt `QSystemTrayIcon`, keeping dependencies light
- If no tray host is available (for example GNOME without the AppIndicator extension), the app continues without a tray, logs a clear message, and `doctor` reports it
- Menu: Pause or Resume detection, Show or Hide preview, Calibrate, Open config file, Quit
- Icon states: active, paused, error (for example camera lost). Use simple bundled icons
- Tooltip shows state and active player backend

### Detection toggle gesture

- Default: both hands showing `Closed_Fist`, held 2000 ms, toggles pause and resume. Configurable
- This requires `num_hands=2` in the recognizer
- When two hands are present, only the toggle gesture is evaluated. All other gestures are ignored. With one hand, behavior is unchanged
- Paused state: the camera stays open at a reduced rate (default 5 fps, `paused_fps`) and only the toggle gesture is evaluated. All actions are suppressed
- Tray option 'Release camera while paused' (default off). When on, resume works only from the tray
- The camera indicator light stays on while paused unless the camera is released. Document this in the README

### Notifications

- Desktop notifications through the freedesktop notification service (D-Bus `org.freedesktop.Notifications`, or `notify-send` as a fallback). If unavailable, log to the terminal and continue
- Send a notification when: the app starts and becomes active, detection is paused, detection is resumed, the camera fails, the player backend is lost
- Each notification shows the new state and the active backend. No sound, auto-expire after about 3 seconds, rate-limit identical notifications to one per 2 seconds

### Config additions

```yaml
tray:
  enabled: true
  release_camera_when_paused: false
pause:
  gesture: two_fists
  hold_ms: 2000
  paused_fps: 5
notifications:
  enabled: true
  on_start: true
  on_toggle: true
  on_error: true
```

### Tests

- Engine: two-fist hold toggles pause, too-short hold does nothing, paused state suppresses all other gestures, single-hand fist mute still works, two hands present suppresses single-hand actions
- Tray and notifier sit behind interfaces with fakes. Test that the right state changes trigger the right calls and that rate limiting works
- `doctor` checks tray host availability and the notification service

### Acceptance

- On start, a notification appears and the tray shows active
- Holding two fists for 2 seconds pauses, shows a notification, and changes the tray icon. Repeating resumes
- Tray Quit exits cleanly and releases the camera
- Missing tray host or notification service never crashes the app

## 20. Amendment: KDE Target and Notify-on-Every-Action

This section overrides conflicting statements in section 19 (Tray icon, Notifications).

### Desktop environment

- Author uses KDE Plasma on Wayland. Plasma supports the StatusNotifierItem tray protocol and the freedesktop notification service natively
- Tray: prefer a StatusNotifierItem implementation that works on Plasma Wayland. The agent still runs the short spike (`QSystemTrayIcon` via Qt, or a lightweight D-Bus SNI library). Pick the option with the smallest dependency footprint that works. GNOME fallback behavior from section 19 stays as is
- `doctor` must verify on Plasma: tray host present, notification service present, `XDG_CURRENT_DESKTOP` and session type reported

### Notification policy: every action

A desktop notification is sent for every state change and every action the app performs.

| Event | Notification text (example) |
| --- | --- |
| App on | Gesture Media Controller started. Active, backend: mpris |
| App off | Gesture Media Controller stopped |
| Detection paused or resumed | Detection paused (two fists) |
| Play/pause | Play/Pause (Open palm) |
| Mute toggle | Mute toggled (Fist). Show the muted state when it is known |
| Next or previous | Next track (Thumb up) |
| Volume | Volume 45% (Pointing up) |
| Seek | Seek +10 s (Victory) |
| Pitch | Pitch +2.5 semitones (ILoveYou) |
| Pitch reset | Pitch reset |
| Open player | Opening player |
| Unsupported action | Pitch not supported on VLC. Show once per capability per backend |
| Errors | Camera lost, player backend lost, tray or notifier failure |
| Calibration | Calibration saved or cancelled |

### Handling continuous actions without spam

- Discrete actions: one notification per fired action
- Continuous actions (volume, seek, pitch): use ONE notification per gesture engagement and update it in place using the freedesktop `replaces_id` mechanism, so the popup shows the live value instead of stacking
- Throttle in-place updates to at most 5 per second. Always send a final update with the last value when the gesture ends
- Auto-expire a continuous notification about 1.5 seconds after its last update. Discrete ones expire after about 3 seconds
- Implementation: use D-Bus `org.freedesktop.Notifications.Notify` directly (for example with `jeepney`), or `notify-send --print-id --replace-id` if the installed libnotify supports it. Keep the notification id between updates
- Replace the identical-notification rate limit in section 19 with this behavior. Unsupported-action notices stay limited to once per capability per backend
- Notifications are silent (no sound) and low urgency, except errors which use normal urgency
- Include the source gesture name in the body, so the user can learn the mapping

### Config

```yaml
notifications:
  enabled: true
  verbosity: all          # all | important | off
  update_hz: 5
  expire_discrete_ms: 3000
  expire_continuous_ms: 1500
  per_action:             # optional overrides, true or false
    volume: true
    seek: true
    pitch: true
```

- `important` limits notifications to app on or off, pause or resume, and errors
- Tray menu gets a 'Notifications' submenu with All, Important only, Off

### Tests

- Notifier interface with a fake. Assert one notification per discrete action
- Continuous gesture produces one notification id that is reused for updates, throttled to `update_hz`, with a final update on release
- Verbosity levels filter correctly. Unsupported-action notice fires once per capability and backend
- App start and stop each produce a notification, including on SIGTERM and tray Quit
- A failing notification service never blocks the frame loop or crashes the app

### Acceptance

- On Plasma, every gesture action produces a visible notification
- Holding volume and moving the hand shows one popup that updates live, with no stacking
- Starting and quitting the app each show a notification

## 21. Amendment: Live Updates Plus Final Value

This section overrides the continuous-action rules in section 20 where they conflict.

### Behavior for volume, seek, and pitch

- While the gesture is active: one notification updates live in place (via `replaces_id`), throttled to `update_hz`. Body shows the live value and the gesture name, for example `Volume 45% (Pointing up)`
- When the gesture ends (release, hand lost beyond grace, or gesture switch): send a final notification that replaces the live one and shows the resulting value, for example `Volume set to 45%`, `Seek total +14 s, now at 1:32`, `Pitch set to +2.5 semitones`
- The final notification uses the discrete expiry (`expire_discrete_ms`, default 3000). Live updates keep `expire_continuous_ms`
- Seek final value: show the total seconds moved during the engagement. Include the new position only when the backend can report it
- If the value did not change during the engagement (for example the gesture stayed inside the deadzone), send no final notification

### Config

```yaml
notifications:
  live_updates: true        # live in-place updates while gesturing
  final_value: true         # final result notification when the gesture ends
```

- Both default to true
- If `live_updates` is false, only the final value is shown. If `final_value` is false, only live updates are shown
- Tray 'Notifications' submenu gets two toggles: Live updates, Final value

### Tests

- Gesture engagement produces N throttled live updates sharing one id, then exactly one final notification on the same id with the resulting value
- No change in value produces no final notification
- Each config combination of `live_updates` and `final_value` behaves as specified
- Final notification is still sent when the hand disappears mid-gesture

### Acceptance

- Moving the hand during volume shows a live popup, and releasing the gesture shows the final volume for about 3 seconds
