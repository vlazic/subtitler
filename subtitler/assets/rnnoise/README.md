# Bundled RNNoise model

`sh.rnnn` is the `somnolent-hogwash-2018-09-01` weight set from
https://github.com/GregorR/rnnoise-models, taken verbatim
(sha256 `70bb6685eb0c2a1d18e2918dca3fbfbd39317010b1802eb1b6ea73a92f3fdec0`, 297646 bytes).

## Licence

Two things are worth keeping straight, because they carry different terms.

- **These weights.** The upstream repository states that, outside its `tools/` directory,
  "none of this work is creative and thus none of it is subject to copyright". No licence
  file is shipped with it, and none is claimed here. Only the weight file is vendored; the
  `tools/` directory is not.
- **The algorithm.** RNNoise itself is Xiph.Org's, under the BSD 3-clause licence, and it
  reaches this project only as ffmpeg's built-in `arnndn` filter. Nothing is compiled or
  vendored for it, which is the whole reason `arnndn` replaced the x86-64 `rnnoise` C
  checkout the prior art carried.

## Why this model of the six

The upstream set is a matrix of expected signal against expected noise.
`somnolent-hogwash` is the speech-signal, recording-environment cell: a person talking,
recorded somewhere with a fan, an air conditioner or a computer in the room. That is the
primary user story exactly, so it is the one that ships. The others target general audio or
non-speech human sound and would be the wrong default here.

## Why it is bundled rather than downloaded

`arnndn` refuses to run without a model file. A preset that appears in `--denoise --help`
and then fails with "the arnndn preset needs an rnnoise model file" is a preset that does
not exist. 291 KB inside the wheel is cheaper than a network dependency at denoise time,
and it keeps the local-first promise: nothing this pipeline needs is fetched mid-run.

## How it is passed to ffmpeg

Never as an absolute path. `arnndn` takes its model as the filter option `m=`, and filter
options are colon separated, so any path containing a colon splits the filtergraph in the
wrong place. `media.denoise_audio` copies this file into a temp directory as
`rnnoise.rnnn` and runs ffmpeg with `cwd` set there against the literal `arnndn=m=rnnoise.rnnn`,
the same strategy `burn.py` uses for `ass=f=subs.ass`.
