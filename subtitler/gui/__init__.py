"""Two graphical front ends, for the person who will never open a terminal.

**The native window is the default and the browser page is the fallback.** The original
reasoning here rejected tkinter outright, and it was half right. Tk is a property of the
interpreter *build*, and Homebrew's `python@3.12` really does ship without `python-tk`; but
this project installs through uv, and uv's python-build-standalone distributions include
Tk 8.6 on both macOS and Linux. So the failure the browser page was avoiding is not the one
the target user hits, and the browser page has a failure of its own that no argument about
imports can fix: the friend still has to open a terminal and type `subtitler gui`, which is
the exact thing a GUI exists to spare them. `subtitler install-app` gives them an icon; an
icon has to open a window.

The browser page stays as the fallback, for a Python whose Tk is genuinely missing and for
a machine with no display, and `cli._cmd_gui` picks between them.

Neither view is allowed to know anything the other does not, and neither contains a rule
about subtitles. The layers, smallest display-dependent piece last, the way `doctor.py`
puts platform facts behind `Platform`:

  * `forms`   - payload to `RunConfig`, and the command line that matches it. Pure.
  * `files`   - the file picker's model, and the one macOS branch (`open` vs `xdg-open`).
  * `jobs`    - background work and its log, with the worker behind an injectable `spawn`.
  * `session` - phases, the cue editor's model, and where corrections are staged. Shared.
  * `app`     - the browser page's routing and JSON, as `handle(...) -> Response`.
  * `server`  - the only module that binds a socket or opens a browser.
  * `window`  - the only module that creates a widget.

Importing this package touches no display, starts no server and imports no Tk, so
`subtitler --help` and the test suite pay nothing for either front end.
"""

from __future__ import annotations

__all__ = ["app", "files", "forms", "jobs", "session"]
