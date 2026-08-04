"""faster-whisper (CTranslate2) adapter.

The default local engine everywhere except Apple Silicon. Most of the non-obvious code
here is lifted from `record-audio/record_audio/transcribe.py`, where it was learned the
expensive way.
"""

from __future__ import annotations

import ctypes
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from subtitler import models
from subtitler.engines.base import (
    Availability,
    EngineUnavailable,
    ModelInfo,
    TranscribeOptions,
    collapse_repetition,
    drop_silent_segments,
    prompt_echoed,
)
from subtitler.model import Segment, Transcript, Word

BACKEND = "faster-whisper"

# A sane batch size on a 24 GB card, measured rather than guessed. On an RTX 3090 decoding
# large-v3 in float16, batch 16 peaks at about 10 GB and 32 at about 22.5 GB while being
# only 6% faster, which leaves no room for a desktop session on the same card.
DEFAULT_BATCH_SIZE = 16

# None = not attempted yet. Cache the RESULT, not merely the fact that it ran: caching a
# bare "already done" flag made the second call return True even when nothing loaded,
# which told the device probe CUDA was fine and produced a failure several seconds later.
_cuda_preloaded: bool | None = None


def _nvidia_lib_dirs(roots: Sequence[str] | None = None) -> list[Path]:
    """Every `nvidia/*/lib` directory pip put in this environment.

    **`__file__` is None for these packages.** `nvidia`, `nvidia.cublas` and
    `nvidia.cublas.lib` are all namespace packages with no `__init__.py`, so the import
    machinery gives them a `__path__` and no `__file__`. The first version of this function
    read `Path(module.__file__ or "").parent`, which quietly became `Path(".")`, so every
    candidate library was looked for in the current working directory, none existed, and
    the preload returned False on a machine where all of them were installed. It never
    loaded a single library, on any machine, ever. `__path__` is the attribute that exists.
    """
    if roots is None:
        try:
            nvidia = __import__("nvidia", fromlist=["__path__"])
        except ImportError:
            return []
        roots = list(getattr(nvidia, "__path__", []))
    return sorted({d for root in roots for d in Path(root).glob("*/lib") if d.is_dir()})


def preload_cuda_libraries() -> bool:
    """Load the venv's CUDA 12 libraries before CTranslate2 looks for them.

    CTranslate2 wheels link against CUDA 12. This machine's system toolkit is 11.5, so the
    loader finds `libcublas.so.11` and nothing that satisfies `libcublas.so.12`, and
    CTranslate2 dies with "Library libcublas.so.12 is not found or cannot be loaded". The
    driver is not the problem: 580 supports 12 and 13 alike. Opening the pip-installed cu12
    libraries with RTLD_GLOBAL registers them under their sonames, so the later `dlopen`
    from inside CTranslate2 and cuDNN finds the already-loaded 12 instead of searching a
    system path that only has 11.

    Everything under `nvidia/*/lib` is loaded rather than a hand-written list of filenames:
    cuDNN 9 is split into sub-libraries (`libcudnn_graph`, `libcudnn_engines_*`) that
    `libcudnn.so.9` dlopens by bare soname at runtime, and nvrtc arrives as a cuDNN
    dependency. Naming them individually means a version bump that renames one silently
    reverts to CPU.

    Returns True if anything was loaded. Never raises: on a CPU-only machine, and on both
    CI runners, there is simply nothing to preload.
    """
    global _cuda_preloaded
    if _cuda_preloaded is not None:
        return _cuda_preloaded

    loaded = False
    for lib_dir in _nvidia_lib_dirs():
        for candidate in sorted(lib_dir.glob("*.so.*")):
            try:
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                loaded = True
            except OSError:
                # A library that will not load is not fatal here: the device probe below
                # falls back to CPU, which is slow but correct.
                continue

    _cuda_preloaded = loaded
    return loaded


def _loadable(soname: str) -> bool:
    try:
        ctypes.CDLL(soname, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return False
    return True


def cublas12_loadable() -> bool:
    """Is `libcublas.so.12` actually openable, after the venv preload has had its turn?

    Asked by name, not by "did the preload load anything": the preload walks a whole
    directory tree and would report success having loaded only nvrtc while cuBLAS itself
    was missing, which is exactly the case CTranslate2 then dies on.
    """
    preload_cuda_libraries()
    return _loadable("libcublas.so.12")


def _cuda_usable() -> bool:
    """A visible device is not enough: the CUDA 12 libraries have to actually load.

    `get_cuda_device_count()` reports the driver's devices, which on this dev box is 1
    even though the system CUDA toolkit is 11.5 and CTranslate2 wants 12. Trusting that
    count alone produces "Library libcublas.so.12 is not found or cannot be loaded" at
    model load time, several seconds later and with no hint about the cause. So: check
    that libcublas.so.12 is genuinely loadable before claiming CUDA.
    """
    if os.environ.get("SUBTITLER_FORCE_CPU"):
        return False
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() <= 0:
            return False
    except Exception:
        return False

    return cublas12_loadable()


def cuda_report() -> dict[str, Any]:
    """Everything `subtitler doctor` needs to explain the GPU decision, as plain JSON.

    Returned rather than printed so `doctor` can run this in a **subprocess**: answering
    the question means dlopening several hundred megabytes of CUDA libraries with
    RTLD_GLOBAL, and a dependency report has no business doing that to its own process
    (nor paying the import cost on a Mac, where the answer is always "no CUDA here").

    `usable` is `_cuda_usable()` itself, not a re-derivation of it, so the doctor can never
    report something the engine then disagrees with.
    """
    report: dict[str, Any] = {
        "ctranslate2": "",  # its version, once we know it is importable
        "devices": 0,
        "packages": [],
        "cublas12": False,
        "cudnn9": False,
        "usable": False,
        "error": "",
    }
    try:
        import ctranslate2

        report["ctranslate2"] = getattr(ctranslate2, "__version__", "unknown")
        report["devices"] = int(ctranslate2.get_cuda_device_count())
    except Exception as exc:  # an import error, or a driver that cannot be queried
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    preload_cuda_libraries()
    report["packages"] = sorted({d.parent.name for d in _nvidia_lib_dirs()})
    report["cublas12"] = _loadable("libcublas.so.12")
    report["cudnn9"] = _loadable("libcudnn.so.9")
    report["usable"] = _cuda_usable()
    return report


class FasterWhisperEngine:
    name = BACKEND
    kind = "local"

    def __init__(
        self, model: str = "large-v3", *, device: str = "auto", batch_size: int = 0
    ) -> None:
        self.spec = models.resolve(model, BACKEND)
        self.requested_device = device
        self.requested_batch_size = max(0, int(batch_size))
        self._model: Any = None
        # Set once the model actually loads, which is the only reliable answer.
        self._resolved: tuple[str, str] | None = None
        self._fallback_reason: str | None = None

    # ---------------------------------------------------------------- availability

    def availability(self) -> Availability:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return Availability(False, "faster-whisper is not installed", "uv sync --extra local")
        if models.local_path(self.spec) is None:
            return Availability(
                False,
                f"the {self.spec.name} weights are not downloaded ({self.spec.size_label})",
                f"subtitler models download {self.spec.name}",
            )
        return Availability(True)

    def resolve_device(self) -> tuple[str, str]:
        """(device, compute_type). int8 on CPU is the difference between usable and not."""
        if self._resolved is not None:
            return self._resolved
        if self.requested_device == "cpu":
            return "cpu", "int8"
        if self.requested_device == "cuda":
            return "cuda", "float16"
        return ("cuda", "float16") if _cuda_usable() else ("cpu", "int8")

    def effective_batch_size(self) -> int:
        """Batching is a GPU throughput trick, so it is silently 0 anywhere else.

        `BatchedInferencePipeline` runs on CPU too and is slower there than the sequential
        path, so honouring the flag after a CUDA fallback would make an already-degraded
        run worse. 0 means "decode sequentially", which is the default everywhere.
        """
        if self.requested_batch_size <= 0:
            return 0
        return self.requested_batch_size if self.resolve_device()[0] == "cuda" else 0

    def ensure_model(self, progress: Any = None) -> ModelInfo:
        path = models.local_path(self.spec) or models.download(self.spec, progress=progress)
        return ModelInfo(
            name=self.spec.name,
            revision=self.spec.revision,
            path=Path(path),
            size_bytes=self.spec.approx_bytes,
        )

    def describe(self) -> dict[str, Any]:
        device, compute = self.resolve_device()
        return {
            "engine": self.name,
            "kind": self.kind,
            "model": self.spec.name,
            "repo": self.spec.repo_id,
            "revision": self.spec.revision,
            "device": device,
            "compute_type": compute,
            "batch_size": self.effective_batch_size(),
            "cuda_fallback": self._fallback_reason,
        }

    # ---------------------------------------------------------------- transcription

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel

        path = models.local_path(self.spec)
        if path is None:
            raise EngineUnavailable(
                self.name,
                "weights are not downloaded",
                f"subtitler models download {self.spec.name}",
            )

        device, compute_type = self.resolve_device()
        if device == "cuda":
            # `resolve_device()` reaches the preload only through `_cuda_usable()`, and it
            # only consults that on `--device auto`. An explicit `--device cuda` therefore
            # went straight to CTranslate2 with nothing preloaded, which looked for
            # libcublas.so.12 against a system toolkit that ships 11.5 and died at the
            # first decoded window, on a machine where `doctor` had just reported CUDA as
            # usable. Found from the GUI, whose Processor dropdown makes "cuda" one click
            # away rather than something only a benchmark run ever typed.
            preload_cuda_libraries()
        self._model = WhisperModel(str(path), device=device, compute_type=compute_type)
        self._resolved = (device, compute_type)
        return self._model

    def _force_cpu(self, reason: str) -> None:
        """Drop to CPU after a CUDA failure, recording why.

        Necessary because CTranslate2 initializes the device lazily: constructing
        WhisperModel with device="cuda" succeeds on a machine with no usable CUDA
        runtime, and the failure only appears once decoding starts. No probe can catch
        that ahead of time, so the retry has to wrap the actual work.
        """
        self._fallback_reason = reason.strip().splitlines()[0] if reason else "cuda unavailable"
        self._model = None
        self._resolved = ("cpu", "int8")

    def transcribe(self, audio: Path, opts: TranscribeOptions) -> Transcript:
        avail = self.availability()
        if not avail.ok:
            raise EngineUnavailable(self.name, avail.reason, avail.fix)

        try:
            return self._decode(audio, opts)
        except Exception as exc:
            # A CUDA runtime that is visible but unusable fails here, not at load time.
            # An explicit --device cuda should report that; auto should degrade to CPU,
            # which is slower but correct, and say so in the params.
            if self._resolved != ("cuda", "float16") or self.requested_device == "cuda":
                raise
            self._force_cpu(str(exc))
            return self._decode(audio, opts)

    @staticmethod
    def _prompt_for(opts: TranscribeOptions, batch_size: int) -> str | None:
        """The steering prompt, unless batching is on, in which case there is none.

        Not a preference: `BatchedInferencePipeline` prepends `initial_prompt` to **every**
        window in the file, because `generate_segment_batched` passes it as
        `previous_tokens` for each batch element and there is no `prompt_reset_since` to
        move past it. The sequential path does move past it, but only *after* the first
        30-second window: `generate_segments` seeds `all_tokens` with the prompt and raises
        `prompt_reset_since` to `len(all_tokens)` at the end of the first iteration when
        `condition_on_previous_text` is off. So window one is exposed on both paths, and
        this dropping the prompt unconditionally is not what protects the sequential one.
        `_decode` handles the exposure that is left; see the echo retry there.

        Measured on a 54-minute Serbian episode: batch 16 with the prompt attached echoed
        "Zadrži srpski jezik i latinično pismo..." back as transcript text over and over and
        came out 900 words (15%) short of the sequential transcript. Without the prompt the
        same run produced 6231 words against sequential's 6186, with no echo at all. So the
        prompt is dropped rather than the batching, and `--batch-size` documents the trade.
        """
        return None if batch_size else opts.initial_prompt

    def _decode(self, audio: Path, opts: TranscribeOptions) -> Transcript:
        """Decode, and decode a second time without the prompt if the first echoed it.

        The prompt is only ever in play for the first 30-second window (see `_prompt_for`),
        which is harmless while that window holds speech and is the whole transcript when
        the clip is shorter than one window. The first 10 seconds of a YouTube episode,
        titles and music and no speech at all, came back as the single cue "Zadrži srpski
        jezik i latinično pismo." with word confidences of 0.02 to 0.11; a different
        `--prompt` produced that prompt instead, which is the mechanism admitting to itself.

        The trigger is the echo, not the duration. A duration threshold would leave the
        long-file half of this bug (`--denoise arnndn` on `uvod-u-pravo.m4a`, where the
        echo replaced the opening of a 164-second lecture) unfixed while taking the
        steering away from every legitimate short clip, and mean word confidence fires on
        merely difficult audio too. The one prompt-free retry cannot echo again, costs
        nothing on the runs that never echo, and keeps the prompt everywhere it works.
        """
        # Checked before anything is imported or loaded, so the message arrives instead of
        # "No clip timestamps found" from deep inside faster-whisper's batched generator,
        # which has nothing to chunk on when the VAD is off.
        if self.effective_batch_size() and not opts.vad:
            raise ValueError(
                "batched decoding needs the VAD to split the audio into chunks; "
                "use --batch-size 0 to decode sequentially without it"
            )

        prompt = self._prompt_for(opts, self.effective_batch_size())
        transcript = self._decode_with(audio, opts, prompt)
        echo_n, echo_text = prompt_echoed(transcript.text, prompt)
        if not echo_n:
            return transcript

        transcript = self._decode_with(audio, opts, None)
        # Recorded rather than logged: an engine has no log channel, and the pipeline reads
        # this back out of `transcript.json` so a cached run reports it too.
        transcript.params["prompt_echo_retry"] = echo_text
        return transcript

    def _decode_with(self, audio: Path, opts: TranscribeOptions, prompt: str | None) -> Transcript:
        import ctranslate2

        # A fixed seed does not make CTranslate2 bit-deterministic, but it removes one
        # source of run-to-run drift when comparing engines in the benchmark.
        ctranslate2.set_random_seed(opts.seed)

        model = self._load()
        batch_size = self.effective_batch_size()
        runner, kwargs = model, {}
        if batch_size:
            from faster_whisper import BatchedInferencePipeline

            runner = BatchedInferencePipeline(model=model)
            kwargs["batch_size"] = batch_size

        started = time.monotonic()
        segments_iter, info = runner.transcribe(
            str(audio),
            language=None if opts.language == "auto" else opts.language,
            initial_prompt=prompt,
            beam_size=opts.beam_size,
            temperature=opts.temperature,
            word_timestamps=opts.word_timestamps,
            # Off by default: this is the main driver of repetition loops, and it lets one
            # bad segment poison everything after it.
            condition_on_previous_text=opts.condition_on_previous_text,
            compression_ratio_threshold=opts.compression_ratio_threshold,
            vad_filter=opts.vad,
            **kwargs,
        )

        segments = []
        # Counted, not just applied: the benchmark reports how often the decoder had to be
        # rescued from a repetition loop, and once the transcript exists the evidence is
        # gone by definition. Same for the silence gate below.
        collapsed = 0
        for seg in segments_iter:  # generator: work happens here
            raw_text = (seg.text or "").strip()
            text = collapse_repetition(raw_text)
            collapsed += text != raw_text
            if not text:
                continue
            segments.append(
                Segment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=text,
                    words=tuple(
                        Word(
                            start=float(w.start),
                            end=float(w.end),
                            text=str(w.word).strip(),
                            prob=float(w.probability) if w.probability is not None else None,
                        )
                        for w in (seg.words or [])
                    ),
                    no_speech_prob=_opt_float(getattr(seg, "no_speech_prob", None)),
                    avg_logprob=_opt_float(getattr(seg, "avg_logprob", None)),
                    compression_ratio=_opt_float(getattr(seg, "compression_ratio", None)),
                )
            )
        runtime = time.monotonic() - started

        kept = drop_silent_segments(tuple(segments), audio)
        device, compute_type = self.resolve_device()

        return Transcript(
            language=getattr(info, "language", None) or opts.language,
            duration=float(getattr(info, "duration", 0.0) or (kept[-1].end if kept else 0.0)),
            segments=kept,
            engine=self.name,
            model=self.spec.name,
            model_revision=self.spec.revision,
            runtime_s=runtime,
            params={
                "language": opts.language,
                "beam_size": opts.beam_size,
                "temperature": opts.temperature,
                "vad": opts.vad,
                "device": device,
                "compute_type": compute_type,
                "batch_size": batch_size,
                # Recorded because it is the one option the engine overrides on its own: a
                # transcript decoded in batches, or re-decoded after an echo, was never
                # steered by the prompt.
                "initial_prompt": bool(prompt),
                "cuda_fallback": self._fallback_reason,
                "seed": opts.seed,
                "repetition_collapsed": collapsed,
                "silence_dropped": len(segments) - len(kept),
            },
        )


def _opt_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
