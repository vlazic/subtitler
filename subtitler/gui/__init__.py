"""A local web GUI, for the person who will never open a terminal.

Why a browser page and not tkinter: **zero compiled dependencies** (non-negotiable 6)
leaves exactly two realistic toolkits, stdlib `tkinter` and a local HTTP server driving the
system browser. This is the second one, for three reasons that all point the same way.

  1. tkinter is a property of the *interpreter build*, not of this project. The python.org
     and uv-managed builds ship Tk; Homebrew's `python@3.12` does not pull `python-tk`, and
     a Python without it fails at `import tkinter` with a message about `_tkinter` that
     means nothing to the audience. `http.server` and `webbrowser` are in every build.
  2. Neither CI runner has a display, and macOS has no Xvfb. A Tk GUI would be entirely
     unexercised on the primary target; every route here is driven by real HTTP in the
     tests, on both runners.
  3. Importing this package touches no display and starts no server, so `subtitler --help`
     and the test suite pay nothing for it.

The parts split so the display-dependent piece is as small as possible, the way `doctor.py`
puts platform facts behind `Platform`:

  * `forms`  - payload to `RunConfig`, and the command line that matches it. Pure.
  * `files`  - the file picker's model, and the one macOS branch (`open` vs `xdg-open`).
  * `jobs`   - background work and its log, with the thread behind an injectable `spawn`.
  * `app`    - routing and JSON, as `handle(method, path, query, body) -> Response`.
  * `server` - the only module that binds a socket or opens a browser.
"""

from __future__ import annotations

__all__ = ["app", "files", "forms", "jobs"]
