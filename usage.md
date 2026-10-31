# Using Gesture Media Controller

Webcam hand gestures control playback on this Linux machine. Recognition stays on the CPU. The app is a terminal process with a preview window, a tray icon, and silent desktop notifications.

Run every command from the repo directory:

```bash
cd ~/code/cv-music-control
source .venv/bin/activate
```

`gesture-media` is also available as `.venv/bin/gesture-media` without activating the environment.

## Check the machine

```bash
gesture-media doctor
```

Each line is `PASS`, `FAIL`, or `WARN`. The process exits 0 when nothing marked `[HARD]` failed. On this machine that means the camera opened, the gesture model is present, MediaPipe imports, and either `playerctl` or libmpv is available. Both are installed now.

`WARN players: no active MPRIS players` is normal until VLC, Spotify, or a browser is actually playing something. `WARN calibration` means you have not saved a profile. The app works without one.

List cameras if the default is the wrong device:

```bash
gesture-media list-cameras
```

## Start controlling media

Start the player first when you want to control an app that is already running.

```bash
# VLC, Spotify, or a browser tab that is playing
gesture-media run
```

`auto` picks MPRIS when `playerctl -l` lists a player. Otherwise it uses the built-in mpv player.

Play local files with mpv:

```bash
gesture-media run --player mpv --media song.mp3
```

Force one backend:

```bash
gesture-media run --player mpris
gesture-media run --player mpv --media song.mp3 another.mp3
```

Other useful flags:

```bash
gesture-media run --camera 1
gesture-media run --no-preview          # tray and terminal only
gesture-media run --no-calibration      # ignore a saved profile
gesture-media run --verbose             # one debug line per frame
gesture-media run --config config.yaml
```

`--camera` and `--source` cannot be combined. `--source` is a video file or a directory of images, used instead of the webcam.

A second `gesture-media run` prints that the app is already running and exits. Quit with `q`, Esc, Ctrl+C, or the tray **Quit** item. That releases the camera and closes the built-in mpv player.

If the preview window does not show under Wayland:

```bash
QT_QPA_PLATFORM=xcb gesture-media run
```

The window title is `Gesture Media Controller`. Keep your hand inside the rectangle on screen. The image is mirrored, so your right is the right side of the picture.

## Gestures

Hold the pose until it fires. Volume and seek work like a joystick: the pose engages, then you move from where your palm started. Up is toward the top of the frame. A small motion inside the deadzone does nothing.

| Pose | What it does |
| --- | --- |
| Closed fist, hold about a third of a second | Play or pause |
| Open palm toward the camera, hold still about a third of a second | Mute or unmute |
| Thumb pointing to your right, hold briefly | Next track |
| Thumb pointing to your left, hold briefly | Previous track |
| Thumb up, keep it up | Volume up. This also unmutes |
| Thumb down, keep it down | Volume down |
| One finger up, then slide right or left | Seek. Right is forward |
| Both hands in fists for about a second and a half | Pause or resume gesture detection |

One hand at a time for everything except the two-fist pause. While both hands are visible, only the two-fist toggle is watched.

After a play, pause, mute, next, or previous action, that gesture will not fire again until you drop it and wait out a short cooldown.

Some players treat "previous" as "restart this track" once you are a few seconds in. The app sends the normal previous command. The player decides what that means.

Volume follows the system output when you are controlling MPRIS, and the mpv volume when you are using `--player mpv`. Change that with `audio.volume_target` in `config.yaml` (`system` or `player`).

## Preview keys

| Key | Action |
| --- | --- |
| `q` or Esc | Quit |
| `d` | Turn detection on or off |
| `h` | Hide or show the overlay |
| `l` | Draw the hand landmarks |

The overlay shows the gesture, score, hold progress, last action, frames per second, volume, mute, and which backend is active. `calibrated profile active` appears after a profile is loaded.

## Tray

The tray icon is green while detection is active, amber while paused, and red on an error such as a lost camera. The tooltip names the state and the player backend.

The menu can pause or resume detection, show or hide the preview, start calibration, open `config.yaml`, set notifications to all, important, or off, toggle live updates and the final-value popup, and quit.

## Pause detection

Hold both fists for about a second and a half. A notification appears and the tray icon turns amber. Do it again to resume.

The camera stays open, at a slower rate, and its indicator light stays on. Only the two-fist gesture is recognized while paused. To drop the camera instead, set this in `config.yaml` and resume from the tray, because the hands are no longer visible:

```yaml
tray:
  release_camera_when_paused: true
```

## Calibrate

Optional. Defaults work without it. The first run mentions this command once and does not require it.

```bash
gesture-media calibrate
gesture-media calibrate --skip-gestures
gesture-media calibrate --camera 1
gesture-media calibrate --reset
```

The window prompts you, in order:

1. Hold a relaxed hand still for 3 seconds.
2. Move to a comfortable top, bottom, left, and right, holding about 1 second at each.
3. Unless you passed `--skip-gestures`, show each of the five poses for about 2 seconds.

Esc cancels and writes nothing. At the end it prints the old and new deadzone, reach, and arming zone, then asks `Save calibration? [y/n]`. Press `y` or `n` in the preview window or in the terminal.

A saved profile lives at `~/.config/gesture-media/profile.yaml` (or under `$XDG_CONFIG_HOME`). It stores numbers only. The next `gesture-media run` uses it. Precedence, from lowest to highest, is built-in defaults, then `config.yaml`, then that profile, then flags such as `--camera` and `--no-calibration`.

## Notifications

Each action posts a silent notification. Volume and seek update a single popup while you hold the pose, then show the final value for a few seconds when you release it. If your hand never left the deadzone, there is no final popup.

The tray can switch this to important events only (start, stop, pause, errors) or off.

## Record and replay

Record landmarks and labels, not images:

```bash
gesture-media record --out run.jsonl
gesture-media replay run.jsonl
```

Replay feeds the file through the gesture engine and a fake player, and prints the actions. It does not move your real volume or player.

## Config

Edit `config.yaml` in the repo. The comments there name the keys. Common changes are the camera index, how long a pose must be held, the deadzone, and which player opens for the "open player" gesture (`player.open_command`, default `vlc`).

## When something fails

- Camera will not open: `gesture-media list-cameras`, then `gesture-media run --camera N`. Quit any other app using the webcam. A failed open exits with code 2.
- No player reacts: start VLC, Spotify, or a browser video, confirm `playerctl -l` prints a name, then run `gesture-media run`. Or use `--player mpv --media file`.
- Preview is blank on Wayland: `QT_QPA_PLATFORM=xcb gesture-media run`.
- Pitch does nothing on VLC or Spotify: switch to `--player mpv`. Pitch is an mpv rubberband filter. Tempo stays constant.
- Gestures flicker: raise `vision.min_score` or `engine.stable_ms` in `config.yaml`, or run `gesture-media calibrate` in the lighting you actually use.
- The camera light stays on after you pause with two fists: that is expected unless `release_camera_when_paused` is true.
