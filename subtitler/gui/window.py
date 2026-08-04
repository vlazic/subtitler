"""The native window. The only module in this project that creates a widget.

Everything it knows is in `session.py`; what is here is layout, event bindings and the
`after()` timer that drains the worker. That division is not tidiness: both CI runners are
headless, so anything decided in this file is decided where no test can reach it.

Three mechanics that homemade Tk apps get wrong, and how they are handled:

* **The main loop is never blocked.** A transcription is minutes of work. It runs on a
  worker thread inside `jobs.JobManager`, and `_tick` reads whole snapshots from
  `Session.poll` every quarter second. No widget is touched from the worker, ever: the
  worker's only output is text appended to a `Job` behind its own lock.
* **Everything is on a grid with weights**, so the window resizes into something usable
  rather than a form stranded in the top-left corner. There is no fixed pixel geometry
  beyond a minimum size and an initial one.
* **Errors land in the window.** A user who double-clicked an icon has no terminal to read
  a traceback in, so a failed run puts its message in the status bar and its traceback in
  the log tab.
"""

from __future__ import annotations

import sys
import tkinter as tk
from collections.abc import Callable, Mapping
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from subtitler import __version__, media
from subtitler.gui import forms
from subtitler.gui import session as session_mod
from subtitler.gui.session import Player, Session

POLL_MS = 250

# Wide enough for the cue table's seven columns, tall enough for the editor's list and its
# detail pane at once. A starting size, not a constraint: every container below has a grid
# weight and the window is resizable in both directions.
START_SIZE = (1080, 760)
MIN_SIZE = (820, 560)

FLAGGED_BG = "#fff0f0"
EDITED_BG = "#eef6ff"

# Every control the settings tab creates, by the key `forms.build_config` reads it under.
#
# It is declared rather than derived because no headless test can build a widget: this list
# is what `tests/test_desktop.py` checks the whole option surface against, so that a `run`
# flag which grows a GUI control in `forms` but not here is caught on a runner with no
# display. `__init__` asserts the widgets actually built match it, which closes the other
# half of the loop on any machine where a window really opens.
CONTROLS: tuple[str, ...] = (
    "input",
    "start",
    "end",
    "out_dir",
    "engine",
    "model",
    "device",
    "lang",
    "denoise",
    "prompt",
    "max_line",
    "max_lines",
    "min_dur",
    "max_dur",
    "max_cps",
    "style_preset",
    "canvas",
    "canvas_color",
    "font_size",
    "font",
    "review",
    "burn",
    "srt_only",
    "soft_mux",
    "fix",
    "fix_model",
    "fix_prompt",
    "fix_batch",
    "fix_workers",
    "fix_temperature",
    "fix_markup",
    "drop_intro_phrases",
    "force",
    "batch_size",
    "verbose",
    "dry_run",
)

_MEDIA_TYPES = [
    (
        "Audio and video",
        "*.mp3 *.m4a *.wav *.flac *.ogg *.opus *.aac *.mp4 *.m4v *.mkv *.mov *.avi *.webm",
    ),
    ("All files", "*"),
]


class Window:
    """One window over one `Session`."""

    def __init__(self, *, session: Session | None = None, root: tk.Misc | None = None) -> None:
        self.session = session or Session()
        self.player = Player()
        # `className` sets the second half of WM_CLASS, which defaults to "Tk" and is what
        # a desktop matches a window against its launcher with. Left alone, the window
        # docks under a generic toolkit icon next to the Subtitler one the `.desktop` entry
        # installed, and `StartupWMClass=Subtitler` matches nothing. Verified with `xprop`.
        self.root = root or tk.Tk(className="Subtitler")
        self.root.title("subtitler")
        self.root.geometry("{}x{}".format(*START_SIZE))
        self.root.minsize(*MIN_SIZE)
        self._style()

        self.vars: dict[str, tk.Variable] = {}
        self._current: int | None = None
        self._commit_timer: str | None = None
        # The dependency report already on screen. `_tick` runs four times a second and the
        # check finishes once; without this the panel would be rebuilt continuously and
        # could not be scrolled.
        self._doctor_drawn: object | None = None

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self._build_header()
        self._build_tabs()
        self._build_status()
        self._build_actions()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self._refresh_buttons()
        # The half of the `CONTROLS` check that needs a real widget. It costs a set
        # comparison once per window and it is the only thing that can notice a control
        # that was renamed in the layout and not in the list the tests read.
        assert set(self.vars) == set(CONTROLS), (
            f"the settings tab and window.CONTROLS disagree: "
            f"{sorted(set(self.vars) ^ set(CONTROLS))}"
        )

    # ------------------------------------------------------------------ chrome

    def _style(self) -> None:
        """Pick a theme, which is the one place appearance depends on the platform.

        macOS ships the `aqua` theme and it is the native one, so it is left alone. On
        Linux ttk's default theme is the 1990s Motif look; `clam` is the stock theme that
        is not. The branch goes through `Platform` like every other platform fact here.
        """
        style = ttk.Style(self.root)
        if not self.session.plat.is_macos and "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Head.TLabel", font=("TkDefaultFont", 15, "bold"))
        style.configure("Hint.TLabel", foreground="#666666")
        style.configure("Bad.TLabel", foreground="#a41414")
        style.configure("Good.TLabel", foreground="#1c6b2a")

    def _build_header(self) -> None:
        bar = ttk.Frame(self.root, padding=(12, 10, 12, 4))
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(1, weight=1)
        ttk.Label(bar, text="subtitler", style="Head.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            bar,
            text=f"{self.session.plat.describe()}  .  v{__version__}",
            style="Hint.TLabel",
        ).grid(row=0, column=2, sticky="e")

    def _build_tabs(self) -> None:
        self.tabs = ttk.Notebook(self.root, padding=(8, 4))
        self.tabs.grid(row=1, column=0, sticky="nsew")
        self.setup_tab = _Scrollable(self.tabs)
        self.cue_tab = ttk.Frame(self.tabs, padding=8)
        self.result_tab = ttk.Frame(self.tabs, padding=8)
        self.machine_tab = ttk.Frame(self.tabs, padding=8)
        self.log_tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(self.setup_tab, text="Settings")
        self.tabs.add(self.cue_tab, text="Cues")
        self.tabs.add(self.result_tab, text="Result")
        self.tabs.add(self.machine_tab, text="This machine")
        self.tabs.add(self.log_tab, text="Log")
        self._build_setup(self.setup_tab.body)
        self._build_cues(self.cue_tab)
        self._build_result(self.result_tab)
        self._build_machine(self.machine_tab)
        self._build_log(self.log_tab)

    def _build_status(self) -> None:
        bar = ttk.Frame(self.root, padding=(12, 4))
        bar.grid(row=2, column=0, sticky="ew")
        bar.columnconfigure(0, weight=1)
        self.status = ttk.Label(bar, text="ready", anchor="w")
        self.status.grid(row=0, column=0, sticky="ew")
        self.progress = ttk.Progressbar(bar, mode="determinate", length=220)
        self.progress.grid(row=0, column=1, sticky="e", padx=(12, 0))

    def _build_actions(self) -> None:
        bar = ttk.Frame(self.root, padding=(12, 4, 12, 12))
        bar.grid(row=3, column=0, sticky="ew")
        bar.columnconfigure(0, weight=1)
        self.message = ttk.Label(bar, text="", anchor="w", wraplength=640)
        self.message.grid(row=0, column=0, sticky="ew")
        self.reveal_btn = ttk.Button(
            bar, text=self.session.reveal_label(), command=self._reveal, state="disabled"
        )
        self.reveal_btn.grid(row=0, column=1, padx=4)
        self.approve_btn = ttk.Button(
            bar, text="Approve and make the video", command=self._approve, state="disabled"
        )
        self.approve_btn.grid(row=0, column=2, padx=4)
        self.start_btn = ttk.Button(bar, text="Start", command=self._start)
        self.start_btn.grid(row=0, column=3, padx=4)

    # ------------------------------------------------------------------ setup tab

    def _build_setup(self, body: ttk.Frame) -> None:
        body.columnconfigure(0, weight=1)
        opts = forms.options()
        d0 = opts["defaults"]
        row = 0

        source = self._section(body, row, "What to subtitle")
        row += 1
        self._entry(source, 0, "input", "File or link", width=54, span=2)
        ttk.Button(source, text="Choose a file...", command=self._pick_input).grid(
            row=0, column=3, padx=(6, 0)
        )
        self._hint(
            source,
            1,
            "Paste a YouTube link here instead, if yt-dlp is installed."
            if opts["fetch_available"]
            else f"Links need yt-dlp: {opts['fetch_hint']}",
        )
        self._entry(source, 2, "start", "Start at", width=12, hint="blank means the beginning")
        self._entry(source, 3, "end", "Stop at", width=12, hint="MM:SS or HH:MM:SS")

        out = self._section(body, row, "Where the result goes")
        row += 1
        self._entry(out, 0, "out_dir", "Folder", width=54, span=2)
        ttk.Button(out, text="Choose a folder...", command=self._pick_out).grid(
            row=0, column=3, padx=(6, 0)
        )
        self._hint(out, 1, "Blank writes next to the file. A link always needs a folder.")

        speech = self._section(body, row, "Speech recognition")
        row += 1
        self._combo(speech, 0, "engine", "Engine", opts["engines"])
        self._entry(speech, 1, "model", "Model", width=20, value=d0["model"])
        self._combo(speech, 2, "device", "Device", opts["devices"])
        self._combo(
            speech,
            3,
            "lang",
            "Language",
            [f"{item['code']} ({item['label']})" for item in opts["languages"]],
            value=f"sr ({opts['languages'][0]['label']})",
        )
        self._hint(speech, 4, "The language is always pinned: Whisper hears Serbian as Croatian.")
        self._combo(speech, 5, "denoise", "Clean up audio", opts["denoisers"])
        self._entry(
            speech,
            6,
            "prompt",
            "Steering prompt",
            width=54,
            span=2,
        )
        self._hint(speech, 7, "Blank uses the tuned Serbian prompt. Non-negotiable 3.")

        shape = self._section(body, row, "Cue shape")
        row += 1
        d = d0
        self._entry(shape, 0, "max_line", "Characters per line", width=8, value=str(d["max_line"]))
        self._entry(shape, 1, "max_lines", "Lines per cue", width=8, value=str(d["max_lines"]))
        self._entry(shape, 2, "min_dur", "Shortest cue (s)", width=8, value=str(d["min_dur"]))
        self._entry(shape, 3, "max_dur", "Longest cue (s)", width=8, value=str(d["max_dur"]))
        self._entry(shape, 4, "max_cps", "Characters per second", width=8, value=str(d["max_cps"]))

        look = self._section(body, row, "How it looks")
        row += 1
        self._combo(look, 0, "style_preset", "Style", opts["style_presets"])
        self._entry(look, 1, "canvas", "Canvas for audio", width=14, value=d["canvas"])
        self._entry(look, 2, "canvas_color", "Background", width=14, value=d["canvas_color"])
        self._entry(look, 3, "font_size", "Font size", width=8, hint="blank scales with the canvas")
        self._entry(look, 4, "font", "Font", width=20, hint="blank uses the bundled Noto Sans")
        # Non-negotiable 7, said where the box is rather than in a document nobody opens.
        self._hint(look, 5, "A system font name renders differently on every machine.")

        what = self._section(body, row, "What to produce")
        row += 1
        self._check(what, 0, "review", "Let me read and correct the subtitles first", True)
        self._check(what, 1, "burn", "Burn the subtitles into a video", True)
        self._check(what, 2, "srt_only", "Subtitle files only, no video", False)
        self._check(what, 3, "soft_mux", "Also mux a switchable subtitle track", False)

        fix = self._section(body, row, "Correct the transcript with an LLM")
        row += 1
        self._check(fix, 0, "fix", "Run the correction pass (needs an API key in .env)", False)
        self._entry(fix, 1, "fix_model", "Model", width=32, value=d["fix_model"])
        self._entry(fix, 2, "fix_prompt", "Prompt", width=32, value=d["fix_prompt"])
        self._entry(fix, 3, "fix_batch", "Cues per request", width=8, value=str(d["fix_batch"]))
        self._entry(
            fix, 4, "fix_workers", "Parallel requests", width=8, value=str(d["fix_workers"])
        )
        # Left blank on purpose: current Claude models answer a request carrying
        # `temperature` with a 400, so "unset" and "0" are not the same thing.
        self._entry(
            fix, 5, "fix_temperature", "Temperature", width=8, hint="blank sends none at all"
        )
        self._combo(fix, 6, "fix_markup", "Markup", opts["markup"], value=d["fix_markup"])
        self._entry(fix, 7, "drop_intro_phrases", "Intro phrases file", width=40, span=2)
        ttk.Button(fix, text="Choose...", command=self._pick_phrases).grid(
            row=7, column=3, padx=(6, 0)
        )

        adv = self._section(body, row, "Advanced")
        row += 1
        self._combo(adv, 0, "force", "Recompute from stage", opts["force_stages"])
        self._entry(adv, 1, "batch_size", "Batch size", width=8, hint="0 decodes sequentially")
        # Said next to the box, because the reason is not guessable: a batched run cannot
        # carry the steering prompt, and the transcript is measurably worse without it.
        self._hint(adv, 2, "Above 0 decodes in batches and drops the steering prompt.")
        self._entry(adv, 3, "verbose", "Extra detail in the log", width=8, hint="0 to 3")
        self._check(adv, 4, "dry_run", "Print the commands and do nothing", False)

    def _section(self, body: ttk.Frame, row: int, title: str) -> ttk.Frame:
        frame = ttk.LabelFrame(body, text=title, padding=(10, 6))
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure(2, weight=1)
        return frame

    def _entry(
        self,
        parent: ttk.Frame,
        row: int,
        key: str,
        label: str,
        *,
        width: int = 24,
        value: str = "",
        hint: str = "",
        span: int = 1,
    ) -> ttk.Entry:
        var = tk.StringVar(value=value)
        self.vars[key] = var
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        # A wide box spans into the hint column instead of widening column 1, which every
        # other row in the section shares: one 54-character entry otherwise pushed every
        # short field's hint half a window to the right.
        entry.grid(row=row, column=1, columnspan=span, sticky="ew", padx=(8, 0), pady=2)
        if hint:
            ttk.Label(parent, text=hint, style="Hint.TLabel").grid(
                row=row, column=1 + span, sticky="w", padx=(8, 0)
            )
        return entry

    def _combo(
        self,
        parent: ttk.Frame,
        row: int,
        key: str,
        label: str,
        values: list[str],
        *,
        value: str | None = None,
    ) -> ttk.Combobox:
        var = tk.StringVar(value=value if value is not None else (values[0] if values else ""))
        self.vars[key] = var
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        box = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=28)
        box.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=2)
        return box

    def _check(
        self, parent: ttk.Frame, row: int, key: str, label: str, value: bool
    ) -> ttk.Checkbutton:
        var = tk.BooleanVar(value=value)
        self.vars[key] = var
        box = ttk.Checkbutton(parent, text=label, variable=var)
        box.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        return box

    def _hint(self, parent: ttk.Frame, row: int, text: str) -> None:
        ttk.Label(parent, text=text, style="Hint.TLabel", wraplength=560).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 4)
        )

    def values(self) -> dict[str, Any]:
        """The controls as the payload `forms.build_config` validates.

        The keys are the page's keys on purpose: one validator, one set of error messages
        and one command-line builder serve both front ends, so a control that exists here
        and not there cannot silently mean something different.
        """
        payload = {key: var.get() for key, var in self.vars.items()}
        payload["lang"] = language_code(str(payload.get("lang", "")))
        return payload

    # ------------------------------------------------------------------ cue tab

    def _build_cues(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=3)
        tab.rowconfigure(3, weight=2)

        self.cue_summary = ttk.Label(tab, text="Run something first: the cues appear here.")
        self.cue_summary.grid(row=0, column=0, sticky="w", pady=(0, 6))

        holder = ttk.Frame(tab)
        holder.grid(row=1, column=0, sticky="nsew")
        holder.columnconfigure(0, weight=1)
        holder.rowconfigure(0, weight=1)
        columns = ("start", "end", "dur", "cps", "chars", "text")
        self.tree = ttk.Treeview(holder, columns=columns, show="tree headings", selectmode="browse")
        self.tree.heading("#0", text="#")
        self.tree.column("#0", width=52, minwidth=40, stretch=False, anchor="e")
        for name, title, width, anchor in (
            ("start", "start", 112, "e"),
            ("end", "end", 112, "e"),
            ("dur", "secs", 60, "e"),
            ("cps", "ch/s", 60, "e"),
            ("chars", "chars", 60, "e"),
            ("text", "text", 520, "w"),
        ):
            self.tree.heading(name, text=title)
            self.tree.column(name, width=width, anchor=anchor, stretch=(name == "text"))
        self.tree.tag_configure("flagged", background=FLAGGED_BG)
        self.tree.tag_configure("edited", background=EDITED_BG)
        self.tree.grid(row=0, column=0, sticky="nsew")
        bar = ttk.Scrollbar(holder, orient="vertical", command=self.tree.yview)
        bar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.bind("<<TreeviewSelect>>", self._select_cue)

        detail = ttk.LabelFrame(tab, text="This cue", padding=(10, 6))
        detail.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        detail.columnconfigure(0, weight=1)
        detail.rowconfigure(1, weight=1)

        tools = ttk.Frame(detail)
        tools.grid(row=0, column=0, sticky="ew")
        tools.columnconfigure(0, weight=1)
        self.cue_where = ttk.Label(tools, text="", style="Hint.TLabel")
        self.cue_where.grid(row=0, column=0, sticky="w")
        self.play_btn = ttk.Button(tools, text="Listen", command=self._play, state="disabled")
        self.play_btn.grid(row=0, column=1, padx=4)
        self.revert_btn = ttk.Button(
            tools, text="Undo my changes", command=self._revert, state="disabled"
        )
        self.revert_btn.grid(row=0, column=2, padx=4)

        self.cue_text = tk.Text(detail, height=3, wrap="word", font="TkTextFont", undo=True)
        self.cue_text.grid(row=1, column=0, sticky="nsew", pady=(6, 4))
        self.cue_text.bind("<KeyRelease>", self._text_typed)
        self.cue_text.configure(state="disabled")

        self.cue_lines = ttk.Label(detail, text="", font="TkFixedFont", justify="left")
        self.cue_lines.grid(row=2, column=0, sticky="w")
        self.cue_problems = ttk.Label(detail, text="", style="Bad.TLabel", justify="left")
        self.cue_problems.grid(row=3, column=0, sticky="w", pady=(4, 0))

    # ------------------------------------------------------------------ result tab

    def _build_result(self, tab: ttk.Frame) -> None:
        """What the run produced, and the one button that puts it in front of the user.

        A list rather than a sentence, because a run makes up to three files and "done"
        with a folder button leaves the person hunting for which of them is the video.
        """
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)

        self.result_summary = ttk.Label(tab, text="Nothing has finished yet.")
        self.result_summary.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))

        self.results = ttk.Treeview(
            tab, columns=("what", "file"), show="headings", selectmode="browse", height=6
        )
        self.results.heading("what", text="what")
        self.results.heading("file", text="file")
        self.results.column("what", width=120, stretch=False)
        self.results.column("file", width=640, stretch=True)
        self.results.grid(row=1, column=0, columnspan=3, sticky="nsew")
        # Double-clicking a row is what everyone tries first, so it does the same thing
        # the button does rather than nothing.
        self.results.bind("<Double-1>", lambda _e: self._reveal_selected())
        self.results.bind("<Return>", lambda _e: self._reveal_selected())

        self.result_show = ttk.Button(
            tab, text="Show this file", command=self._reveal_selected, state="disabled"
        )
        self.result_show.grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Button(tab, text=self.session.reveal_label(), command=self._reveal).grid(
            row=2, column=1, sticky="w", padx=6, pady=(8, 0)
        )

        where = ttk.LabelFrame(tab, text="The same run, as a command", padding=(10, 6))
        where.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        where.columnconfigure(0, weight=1)
        # Read-only but selectable: the point of printing the command is that a bug report
        # from the least technical user arrives as a command rather than a description of
        # which boxes were ticked, and a Label cannot be copied out of.
        self.command_box = tk.Text(where, height=3, wrap="word", font="TkFixedFont")
        self.command_box.grid(row=0, column=0, sticky="ew")
        self.command_box.configure(state="disabled")

    def _fill_results(self) -> None:
        self.results.delete(*self.results.get_children())
        outputs = self.session.outputs()
        for position, (label, path) in enumerate(outputs):
            self.results.insert("", "end", iid=str(position), values=(label, str(path)))
        if outputs:
            self.results.selection_set("0")
        self.result_show.configure(state="normal" if outputs else "disabled")
        self.result_summary.configure(
            text=f"{len(outputs)} file{'s' if len(outputs) != 1 else ''} in {self.session.out_dir}"
            if outputs
            else "The run finished but wrote no file this window can show."
        )
        self._show_command()

    def _show_command(self) -> None:
        self.command_box.configure(state="normal")
        self.command_box.delete("1.0", "end")
        self.command_box.insert("1.0", self.session.command)
        self.command_box.configure(state="disabled")

    def _reveal_selected(self) -> None:
        picked = self.results.selection()
        outputs = self.session.outputs()
        if not picked:
            return
        position = int(picked[0])
        if position < len(outputs):
            problem = self.session.reveal(outputs[position][1])
            if problem:
                self._say(problem, bad=True)

    # ------------------------------------------------------------------ machine tab

    def _build_machine(self, tab: ttk.Frame) -> None:
        """`doctor` and the model weights, for the user who has no terminal to run them in.

        Both are things the CLI answers with a command, and a command is a dead end for
        somebody who was handed an icon. `doctor` is the more important of the two: the
        single most common way this project fails on a Mac is an ffmpeg built without
        libass, and the only sign of it is a burn that fails at the very end of a long run.
        """
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)

        head = ttk.Frame(tab)
        head.grid(row=0, column=0, sticky="ew")
        head.columnconfigure(0, weight=1)
        self.doctor_state = ttk.Label(head, text="checking...", anchor="w")
        self.doctor_state.grid(row=0, column=0, sticky="ew")
        ttk.Button(head, text="Check again", command=self._recheck).grid(row=0, column=1)

        holder = ttk.Frame(tab)
        holder.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        holder.columnconfigure(0, weight=1)
        holder.rowconfigure(0, weight=1)
        self.doctor_text = tk.Text(holder, wrap="none", font="TkFixedFont", height=14)
        self.doctor_text.grid(row=0, column=0, sticky="nsew")
        bar = ttk.Scrollbar(holder, orient="vertical", command=self.doctor_text.yview)
        bar.grid(row=0, column=1, sticky="ns")
        self.doctor_text.configure(yscrollcommand=bar.set, state="disabled")

        weights = ttk.LabelFrame(tab, text="Speech models on this machine", padding=(10, 6))
        weights.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        weights.columnconfigure(0, weight=1)
        self.model_list = ttk.Treeview(
            weights,
            columns=("model", "size", "state"),
            show="headings",
            selectmode="browse",
            height=4,
        )
        for name, title, width in (
            ("model", "model", 220),
            ("size", "size", 90),
            ("state", "", 260),
        ):
            self.model_list.heading(name, text=title)
            self.model_list.column(name, width=width, stretch=(name == "state"))
        self.model_list.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.download_btn = ttk.Button(weights, text="Download this model", command=self._download)
        self.download_btn.grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(
            weights,
            text=f"They are for {self.session.backend} and are downloaded once.",
            style="Hint.TLabel",
        ).grid(row=1, column=1, sticky="e", pady=(6, 0))
        self._fill_models()

    def _fill_models(self) -> None:
        self.model_list.delete(*self.model_list.get_children())
        for row in self.session.model_rows():
            self.model_list.insert(
                "",
                "end",
                iid=row["name"],
                values=(
                    row["name"],
                    row["size"],
                    "on this machine" if row["cached"] else "not downloaded",
                ),
            )
        wanted = str(self.vars["model"].get()).strip()
        if wanted and self.model_list.exists(wanted):
            self.model_list.selection_set(wanted)

    def _recheck(self) -> None:
        self.session.start_doctor(refresh=True)
        self.doctor_state.configure(text="checking...")
        self._paint_doctor(force=True)

    def _paint_doctor(self, *, force: bool = False) -> None:
        """Copy the report into the panel once it has one. Called from the timer.

        `_doctor_drawn` is what stops it rewriting the same text four times a second: the
        check finishes once, and a Text widget rebuilt on every tick cannot be scrolled.
        """
        report = self.session.doctor_report
        if report is None:
            return
        if self._doctor_drawn is report and not force:
            return
        self._doctor_drawn = report
        blocking = report.get("blocking") or []
        self.doctor_state.configure(
            text=(
                f"{len(blocking)} required dependencies missing: {', '.join(blocking)}"
                if blocking
                else "everything this needs is installed"
            ),
            style="Bad.TLabel" if blocking else "Good.TLabel",
        )
        self.doctor_text.configure(state="normal")
        self.doctor_text.delete("1.0", "end")
        self.doctor_text.insert("1.0", str(report.get("text") or ""))
        self.doctor_text.configure(state="disabled")

    def _download(self) -> None:
        picked = self.model_list.selection()
        name = picked[0] if picked else str(self.vars["model"].get()).strip()
        if not name:
            self._say("pick a model to download first", bad=True)
            return
        refusal = self.session.download_model(name)
        if refusal is not None:
            self._refuse(refusal)
            return
        self._reset_run_view()
        self.tabs.select(self.log_tab)
        self._refresh_buttons()

    # ------------------------------------------------------------------ log tab

    def _build_log(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        self.log = tk.Text(tab, wrap="none", font="TkFixedFont", height=10)
        self.log.grid(row=0, column=0, sticky="nsew")
        bar = ttk.Scrollbar(tab, orient="vertical", command=self.log.yview)
        bar.grid(row=0, column=1, sticky="ns")
        across = ttk.Scrollbar(tab, orient="horizontal", command=self.log.xview)
        across.grid(row=1, column=0, sticky="ew")
        self.log.configure(yscrollcommand=bar.set, xscrollcommand=across.set, state="disabled")

    def _append_log(self, lines: list[str]) -> None:
        if not lines:
            return
        self.log.configure(state="normal")
        self.log.insert("end", "\n".join(lines) + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # ------------------------------------------------------------------ actions

    def _pick_input(self) -> None:
        chosen = filedialog.askopenfilename(title="Choose audio or video", filetypes=_MEDIA_TYPES)
        if chosen:
            self.vars["input"].set(chosen)

    def _pick_out(self) -> None:
        chosen = filedialog.askdirectory(title="Where should the result go?")
        if chosen:
            self.vars["out_dir"].set(chosen)

    def _pick_phrases(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Phrases to drop from the start of the transcript",
            filetypes=[("Text", "*.txt"), ("All files", "*")],
        )
        if chosen:
            self.vars["drop_intro_phrases"].set(chosen)

    def _start(self) -> None:
        self._say("")
        refusal = self.session.start(self.values())
        if refusal is not None:
            self._refuse(refusal)
            return
        self._reset_run_view()
        self.tabs.select(self.log_tab)
        self._refresh_buttons()

    def _approve(self) -> None:
        self._commit_text(now=True)
        refusal = self.session.approve()
        if refusal is not None:
            self._refuse(refusal)
            return
        self._reset_run_view()
        self.tabs.select(self.log_tab)
        self._refresh_buttons()

    def _reset_run_view(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.progress.configure(maximum=max(len(self.session.stages), 1), value=0)
        self._say("")

    def _refuse(self, refusal: session_mod.Refusal) -> None:
        self._say(str(refusal), bad=True)
        if refusal.field and refusal.field != "cues":
            self.tabs.select(self.setup_tab)
        elif refusal.field == "cues":
            self.tabs.select(self.cue_tab)

    def _reveal(self) -> None:
        target = self.session.out_dir
        outputs = self.session.outputs()
        if outputs:
            target = outputs[0][1]
        if target is None:
            return
        problem = self.session.reveal(Path(target))
        if problem:
            self._say(problem, bad=True)

    def _say(self, text: str, *, bad: bool = False) -> None:
        self.message.configure(text=text, style="Bad.TLabel" if bad else "TLabel")

    # ------------------------------------------------------------------ the editor

    def _fill_editor(self) -> None:
        editor = self.session.editor
        # Cleared before the rows go in: inserting a selection fires `<<TreeviewSelect>>`,
        # which would otherwise try to save the text box into the previous run's cue.
        self._current = None
        self.tree.delete(*self.tree.get_children())
        if editor is None:
            return
        for index in editor.order:
            self.tree.insert("", "end", iid=str(index), text=str(index))
            self._paint_row(index)
        self.cue_summary.configure(text=editor.summary())
        first = editor.flagged() or list(editor.order)
        if first:
            self.tree.selection_set(str(first[0]))
            self.tree.see(str(first[0]))

    def _paint_row(self, index: int) -> None:
        editor = self.session.editor
        if editor is None:
            return
        row = editor.row(index)
        tags = []
        if row["problems"]:
            tags.append("flagged")
        if editor.is_edited(index):
            tags.append("edited")
        self.tree.item(
            str(index),
            values=(
                media.format_timecode(row["start"]),
                media.format_timecode(row["end"]),
                f"{row['duration']:.1f}",
                "-" if row["cps"] is None else f"{row['cps']:.1f}",
                row["chars"],
                " / ".join(row["lines"]) or "(empty)",
            ),
            tags=tags,
        )

    def _select_cue(self, _event: object = None) -> None:
        self._commit_text(now=True)
        picked = self.tree.selection()
        if not picked:
            return
        index = int(picked[0])
        self._current = index
        editor = self.session.editor
        if editor is None or index not in editor.origin:
            return
        self.cue_text.configure(state="normal")
        self.cue_text.delete("1.0", "end")
        self.cue_text.insert("1.0", editor.texts[index])
        self.play_btn.configure(
            state="normal" if self.player.available and self.session.media_path else "disabled"
        )
        self.revert_btn.configure(state="normal" if editor.is_edited(index) else "disabled")
        self._show_detail(index)

    def _show_detail(self, index: int) -> None:
        editor = self.session.editor
        if editor is None:
            return
        row = editor.row(index)
        self.cue_where.configure(
            text="cue {}  .  {} to {}  .  {:.1f}s".format(
                index,
                media.format_timecode(row["start"]),
                media.format_timecode(row["end"]),
                row["duration"],
            )
        )
        self.cue_lines.configure(
            text="\n".join(
                f"{width:>3}  {line}"
                for width, line in zip(row["line_widths"], row["lines"], strict=False)
            )
            or "  (nothing)"
        )
        # `lint_cues`' own wording, unedited: a cue marked in this window is a cue the
        # `lint` command reports, with the same sentence.
        self.cue_problems.configure(text="\n".join(row["problems"]))
        self.cue_summary.configure(text=editor.summary())

    def _text_typed(self, _event: object = None) -> None:
        """Re-lint shortly after the last keystroke, not on every one.

        Every keystroke would re-wrap and repaint a row per character typed, which on a
        long cue is visible as lag. A quarter second after the user stops is not.
        """
        if self._commit_timer is not None:
            self.root.after_cancel(self._commit_timer)
        self._commit_timer = self.root.after(250, lambda: self._commit_text(now=True))

    def _commit_text(self, *, now: bool = False) -> None:
        if self._commit_timer is not None:
            self.root.after_cancel(self._commit_timer)
            self._commit_timer = None
        editor = self.session.editor
        index = self._current
        if editor is None or index is None or index not in editor.origin or not now:
            return
        typed = self.cue_text.get("1.0", "end-1c")
        editor.set_text(index, typed)
        self._paint_row(index)
        self._show_detail(index)
        self.revert_btn.configure(state="normal" if editor.is_edited(index) else "disabled")

    def _revert(self) -> None:
        editor = self.session.editor
        index = self._current
        if editor is None or index is None:
            return
        editor.revert(index)
        self.cue_text.delete("1.0", "end")
        self.cue_text.insert("1.0", editor.texts[index])
        self._paint_row(index)
        self._show_detail(index)
        self.revert_btn.configure(state="disabled")

    def _play(self) -> None:
        editor = self.session.editor
        index = self._current
        source = self.session.media_path
        if editor is None or index is None or source is None:
            return
        row = editor.row(index)
        problem = self.player.play(source, row["start"], row["end"])
        if problem:
            self._say(problem, bad=True)

    # ------------------------------------------------------------------ the timer

    def _tick(self) -> None:
        """Drain the worker and repaint. The only place a job's output reaches a widget."""
        snapshot = self.session.poll()
        if snapshot is not None:
            self._append_log(snapshot["lines"])
            self._show_progress(snapshot)
            kind = self.session.take_finished()
            if kind == "run":
                self._finished()
            elif kind is not None:
                self._downloaded()
        self._paint_doctor()
        self.root.after(POLL_MS, self._tick)

    def _show_progress(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("kind") == "download":
            self._show_download(snapshot)
            return
        stages = list(snapshot["stages"]) or list(self.session.stages)
        stage = snapshot.get("stage")
        done = stages.index(stage) + 1 if stage in stages else 0
        self.progress.configure(maximum=max(len(stages), 1), value=done)
        if snapshot["status"] == "running":
            where = f"{stage} ({done} of {len(stages)})" if stage else "starting"
            self.status.configure(
                text=f"{self.session.label}  .  {where}  .  {snapshot['elapsed_s']:.0f}s"
            )

    def _show_download(self, snapshot: Mapping[str, Any]) -> None:
        """A download has no stages, so the bar measures the bytes that have landed."""
        fraction = self.session.download_fraction()
        self.progress.configure(maximum=100, value=round((fraction or 0.0) * 100))
        if snapshot["status"] == "running":
            share = f"{fraction * 100:.0f}%" if fraction is not None else "downloading"
            self.status.configure(
                text=f"{self.session.label}  .  {share}  .  {snapshot['elapsed_s']:.0f}s"
            )

    def _finished(self) -> None:
        phase = self.session.phase
        if phase == session_mod.FAILED:
            self.progress.configure(value=0)
            self.status.configure(text="it did not finish")
            self._say(self.session.error, bad=True)
            self._append_log([self.session.detail] if self.session.detail else [])
            self.tabs.select(self.log_tab)
        elif phase == session_mod.REVIEW:
            self.status.configure(text="read the cues, correct anything wrong, then approve")
            self._fill_editor()
            self._show_command()
            self.tabs.select(self.cue_tab)
            self._warn_or_say("")
        else:
            self.status.configure(text="done")
            self._fill_results()
            names = ", ".join(f"{label}: {path.name}" for label, path in self.session.outputs())
            self._warn_or_say(names or "finished with no files to show")
            self.tabs.select(self.result_tab)
        self._refresh_buttons()

    def _warn_or_say(self, message: str) -> None:
        """A run's warnings outrank whatever else the message bar was going to hold.

        The one that exists says the transcript may be the steering prompt rather than the
        speech, which the reader has to see before a list of files convinces them the run
        went well. They go into the log too, so they survive the next click.
        """
        warnings = self.session.warnings
        self._append_log([f"warning: {text}" for text in warnings])
        self._say("  ".join(warnings) if warnings else message, bad=bool(warnings))

    def _downloaded(self) -> None:
        """A set of weights arrived, or did not. Never touches the run's phase.

        The model list is refilled rather than the row edited, because a download can fail
        halfway and `models.local_path` is the only honest answer about what is on disk.
        """
        self.progress.configure(value=0)
        self._fill_models()
        if self.session.error:
            self.status.configure(text="the download did not finish")
            self._say(self.session.error, bad=True)
        else:
            self.status.configure(text="the model is on this machine")
            self._say(f"{self.session.label} is downloaded")
            self.tabs.select(self.machine_tab)
        self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        busy = self.session.busy
        self.start_btn.configure(state="disabled" if busy else "normal")
        self.approve_btn.configure(
            state="normal"
            if (self.session.phase == session_mod.REVIEW and not busy)
            else "disabled"
        )
        self.reveal_btn.configure(
            state="normal" if (self.session.out_dir and not busy) else "disabled"
        )
        self.download_btn.configure(state="disabled" if busy else "normal")

    # ------------------------------------------------------------------ lifetime

    def _close(self) -> None:
        if self.session.busy and not messagebox.askokcancel(
            "subtitler", "A run is still going. Close anyway?"
        ):
            return
        self.player.stop()
        self.root.destroy()

    def run(self) -> int:
        # Started here rather than in `__init__` so that constructing a window costs no
        # subprocesses, and started at all because the answer takes seconds to compute and
        # is wanted before the user goes looking for it.
        self.session.start_doctor()
        self.root.after(POLL_MS, self._tick)
        self.root.mainloop()
        return 0


class _Scrollable(ttk.Frame):
    """A frame whose contents scroll vertically, because the settings do not fit.

    Tk has no scrollable frame: only a `Canvas` scrolls, so the real content lives in a
    frame placed inside one. `body` is what callers put widgets in. The two bindings are
    what make it behave: the inner frame's size drives the scroll region, and the canvas's
    width is pushed onto the window so a resize widens the form instead of leaving it at
    its requested width with a horizontal gap.
    """

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        bar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=bar.set)
        self.body = ttk.Frame(self.canvas, padding=(10, 8))
        self._window = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>", lambda e: self.canvas.itemconfigure(self._window, width=e.width)
        )
        # Button-4/5 are X11's wheel; Windows and macOS send <MouseWheel>. Bound on the
        # canvas rather than globally so the cue list keeps its own wheel behaviour.
        self.canvas.bind_all("<Button-4>", self._wheel, add="+")
        self.canvas.bind_all("<Button-5>", self._wheel, add="+")
        self.canvas.bind_all("<MouseWheel>", self._wheel, add="+")

    def _wheel(self, event: tk.Event) -> None:
        if not str(event.widget).startswith(str(self.canvas)):
            return
        step = 0
        if getattr(event, "num", 0) == 4:
            step = -1
        elif getattr(event, "num", 0) == 5:
            step = 1
        elif getattr(event, "delta", 0):
            step = -1 if event.delta > 0 else 1
        if step:
            self.canvas.yview_scroll(step, "units")


def language_code(value: str) -> str:
    """`"sr (Serbian)"` back to `"sr"`.

    The combobox shows the name because a friend picking a language does not know the ISO
    code, and `RunConfig.language` is passed to Whisper verbatim, so the label has to come
    off before it leaves the window. It is a function rather than a lambda so the test that
    proves "Serbian" never reaches the engine can name it.
    """
    return value.strip().split(" ", 1)[0] if value.strip() else ""


def run_window(
    *, session: Session | None = None, factory: Callable[[], Window] | None = None
) -> int:
    """Open the window, or explain in one sentence why it could not open.

    A `TclError` here is almost always "no display": a Tk build with no X server or no
    WindowServer to talk to. The caller (`cli._cmd_gui`) turns a non-zero return into the
    browser fallback, because a user who typed `subtitler gui` wants a GUI, not a stack
    trace about `:0`.

    The return value is therefore load-bearing and not merely an exit code: 0 means a
    window opened and has since been closed, and anything else means the caller should try
    the other front end.
    """
    try:
        window = (factory or (lambda: Window(session=session)))()
    except tk.TclError as exc:
        print(f"the window could not open ({exc})", file=sys.stderr)
        return 1
    return window.run()
