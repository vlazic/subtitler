# Task: adjudicate a reference transcript for `gozba-sample`

Your role definition is `.claude/agents/ref-adjudicator.md`. Read it first; it is the part of
this task that does not change per clip.

- Input: `agents/inputs/adjudicate-gozba-sample.md`
- Output: write JSON, and nothing else, to `agents/outputs/adjudicate-gozba-sample.json`
- Windows to cover: 7, indices 0 to 6

Produce exactly this shape:

```json
{
  "role": "ref-adjudicator",
  "clip": "<clip id>",
  "windows": [{"index": <int>, "text": "<verbatim Serbian for that window>"}],
  "spans": [
    {
      "start": <seconds, float>,
      "end": <seconds, float>,
      "chosen": "<the reading you put in the reference>",
      "candidates": [{"source": "<engine label>", "text": "<that engine's reading>"}],
      "reason": "<why, in one sentence>",
      "confidence": "low" | "medium"
    }
  ],
  "notes": "<anything a human reviewer should know, or an empty string>"
}
```

Hard requirements, each of which makes the output invalid if broken:

1. `windows` must contain **every** index in the input, exactly once, in ascending order.
   A window whose engines all said nothing gets an empty string, not a missing entry.
2. Transcribe, do not edit. The reference must be what the speaker **said**, including a
   false start, a repeated word or an ungrammatical sentence. If the engines agree on
   something clumsy, the clumsy version is the reference. Writing the corrected sentence
   would make every raw engine look wrong and the `--fix` pass look right, which would
   invalidate the benchmark this reference exists to serve.
3. Every place where the engines disagree in a way you could not resolve with certainty from
   the language alone goes in `spans`, with all the candidate readings. Under-flagging is the
   expensive mistake here: a flagged span costs a human thirty seconds with the audio, an
   unflagged wrong one silently moves every WER in the table.
4. Serbian Latin script, the speaker's own dialect and word order, ordinary punctuation.
5. No commentary outside the JSON.
