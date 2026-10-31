# Gesture Media Controller

Webcam hand gestures control media playback on Linux. Recognition runs locally on the CPU. The app is a terminal process with an optional OpenCV preview, a StatusNotifierItem tray icon, and silent desktop notifications.

This machine (Fedora, KDE Plasma, Wayland) already has `wpctl`, `mpv` 0.41 with the rubberband filter, `ffmpeg`, and `notify-send`. **`playerctl` is not installed. `mpv-libs` is not installed, so `import mpv` fails until that package is installed.** System volume still works through `wpctl`. MPRIS control and in-process mpv both need the packages below.

## System packages

Fedora:

```bash
sudo dnf install playerctl mpv-libs wireplumber mpv
```

Debian and Ubuntu:

```bash
sudo apt install playerctl wireplumber mpv libmpv2
```

`mpv` must be built with the rubberband audio filter (Fedora's `mpv` package is). `mpv-libs` / `libmpv2` is the library behind `import mpv`. `playerctl` controls VLC, Spotify, and browsers over MPRIS. `wireplumber` provides `wpctl`.

Do not pip-install `opencv-python`. MediaPipe already installs `opencv-contrib-python`, and having both breaks the import.

## Python

Python 3.12, managed by [uv](https://docs.astral.sh/uv/). MediaPipe wheels do not reliably cover 3.13.

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
uv run python scripts/fetch_model.py
```

`scripts/fetch_model.py` downloads `models/gesture_recognizer.task` (gitignored). Check the setup:

```bash
uv run gesture-media doctor
```

`doctor` prints `PASS`, `FAIL`, or `WARN` for each check. Lines marked `[HARD]` are the ones that can make it exit 1: a missing model file, MediaPipe not importable, or neither `playerctl` nor libmpv available. Camera open failures are soft, so a headless run can still exit 0 when `/dev/video*` exists but cannot be read. On this machine, expect `playerctl` and `libmpv` to fail until the packages above are installed. If `wpctl` and the `mpv` binary are present, rubberband should pass.

## Gestures

Hold a gesture inside the arming zone until it fires. Palm up is a smaller image-y.

| Gesture | Action | Default timing |
| --- | --- | --- |
| Open palm | Play/pause | hold 500 ms |
| Closed fist | Mute toggle | hold 500 ms |
| Thumb up | Next track | hold 400 ms |
| Thumb down | Previous track | hold 400 ms |
| Pointing up | Volume. Up is louder, down is quieter. Hold still to open the player | deadzone 0.05, max 30%/s, still 1500 ms |
| Victory | Seek. Right is forward, left is back | deadzone 0.05, max 20 s/s |
| I love you | Pitch. Up is higher, down is lower. Hold still to reset pitch | deadzone 0.05, max 4 semitones/s, still 1500 ms |
| Both fists | Pause or resume detection | hold 2000 ms |

Keys in the preview: `q` or Esc quit, `d` toggle detection, `h` toggle the HUD, `l` toggle landmarks.

## Run

```bash
uv run gesture-media run
uv run gesture-media run --player mpv --media song.mp3
uv run gesture-media run --player mpris --no-preview
uv run gesture-media run --camera 0 --source clip.mp4   # one of --camera or --source
uv run gesture-media list-cameras
uv run gesture-media record --out run.jsonl
uv run gesture-media replay run.jsonl
```

`--player auto` uses MPRIS when `playerctl -l` lists a player, otherwise the built-in mpv player. `--no-preview` skips the OpenCV window. The tray still starts when a StatusNotifierItem host is available (KDE Plasma on Wayland is). `--no-calibration` ignores the saved profile. The first run with no profile prints one line that `gesture-media calibrate` exists; calibration is never required.

A second `run` sees the lock file in `$XDG_RUNTIME_DIR` (or `~/.cache/gesture-media`) and exits 1. Quit with `q`, Esc, Ctrl+C, SIGTERM, or the tray **Quit** item. That releases the camera, closes mpv, and drains the action worker.

If the preview window does not appear under Wayland, force XWayland:

```bash
QT_QPA_PLATFORM=xcb uv run gesture-media run
```

## Calibrate

Optional. About a minute in the preview window: rest, comfortable top/bottom/left/right, then each gesture (skip with `--skip-gestures`). Esc cancels and writes nothing. The summary prints the old and new numbers and asks `y` or `n` (preview keys, and stdin while the window is open).

```bash
uv run gesture-media calibrate
uv run gesture-media calibrate --camera 1 --skip-gestures
uv run gesture-media calibrate --reset
```

The profile is `~/.config/gesture-media/profile.yaml` (`$XDG_CONFIG_HOME` when that is set). It stores deadzone, max displacement, the arming zone, and a version. No images.

Config precedence, lowest to highest: built-in defaults, `config.yaml` next to the program, the calibration profile, then CLI flags.

## Pitch, rubberband, and vinyl mode

Pitch is an mpv feature. The scale is `2 ** (semitones / 12)`. The player sets the audio filter `@rb:rubberband=pitch-scale=1` and then `af-command set-pitch`, which changes pitch without resetting the graph. Playback speed stays 1, so tempo does not change.

If `mpv --af=help` does not list rubberband, pitch falls back to vinyl mode: `speed` changes, so pitch and tempo move together, and the HUD shows vinyl mode. `doctor` reports the missing filter. MPRIS players do not get pitch; the HUD says pitch is unsupported on that backend, once per capability per backend.

## Camera and pause

Two closed fists held for 2 seconds pause detection. Repeat to resume. While paused the camera stays open at `pause.paused_fps` (default 5) and only the two-fist toggle is evaluated. **The camera indicator light stays on while paused.** Set `tray.release_camera_when_paused: true` to release the device instead; resume then works only from the tray, because the gesture is no longer visible.

`--verbose` logs one debug line per frame. Frames are not written to disk. `record` writes landmarks and labels as JSONL and nothing else.

## Players

Volume defaults to the system sink (`wpctl`) for MPRIS and to the mpv player volume for the mpv backend, unless `audio.volume_target` is set explicitly.

Some MPRIS players treat `previous` as "restart the current track" when playback is more than a few seconds in. That is the player's behavior; this app sends `playerctl previous`.

`player.routing` can send one capability to a named backend, for example pitch to mpv and transport to mpris. An empty map keeps transport on the selected player and system volume on `wpctl`.

## Notifications and tray

Every action can post a silent, low-urgency notification (errors use normal urgency). Volume, seek, and pitch reuse one notification id and update it in place, throttled to `notifications.update_hz` (default 5 per second). Releasing the gesture sends one final value on that same id ("Volume set to 45%", "Seek total +14 s", "Pitch set to +2.5 semitones") unless the value never changed. `live_updates: false` keeps only the final. `final_value: false` keeps only the live updates. `verbosity: important` keeps app on/off, pause/resume, and errors. `off` sends nothing.

The tray menu can pause detection, show or hide the preview, start calibration, open the config file, switch notification verbosity, toggle live updates and the final value, and quit. If the tray host is missing the app logs that and keeps running.

## Optional systemd user unit

Not required. Nothing in the app starts itself.

`~/.config/systemd/user/gesture-media.service`:

```ini
[Unit]
Description=Gesture Media Controller
PartOf=graphical-session.target

[Service]
WorkingDirectory=%h/code/cv-music-control
ExecStart=%h/code/cv-music-control/.venv/bin/gesture-media run --no-preview
Restart=on-failure
Environment=QT_QPA_PLATFORM=xcb

[Install]
WantedBy=graphical-session.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now gesture-media.service
```

## Tests

```bash
uv run pytest
uv run ruff check
```

The mpv pitch integration test is skipped when `import mpv` fails or rubberband is missing. The recognizer smoke test is skipped until `tests/fixtures/gestures.mp4` and `labels.json` exist (those are recorded by hand). Replay skips if the engine module is not installed yet.
