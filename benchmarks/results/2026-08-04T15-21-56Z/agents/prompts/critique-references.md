# Task: find what the adjudicator got wrong

Your role definition is `.claude/agents/ref-critic.md`. Read it first.

- Input: `agents/inputs/critique-references.md`
- Output: write JSON, and nothing else, to `agents/outputs/critique-references.json`
- Clips under review: gozba-sample, uvod-u-pravo

Produce exactly this shape:

```json
{
  "role": "ref-critic",
  "findings": [
    {
      "clip": "<clip id>",
      "start": <seconds, float>,
      "end": <seconds, float>,
      "issue": "smoothed" | "grammar_corrected" | "other",
      "reference_text": "<what the reference says there>",
      "candidates": ["<engine reading>", "..."],
      "recommendation": "<what a human should check, in one sentence>",
      "severity": "high" | "medium" | "low"
    }
  ],
  "verdict": "<one paragraph: is this reference fit to score against, and where is it weakest>"
}
```

You are adversarial. The two failures worth your attention:

1. **Silent grammar correction.** The reference reads as better Serbian than any engine
   produced. That is not adjudication, it is editing, and a reference edited that way
   penalises every raw engine and rewards the correction pass under test.
2. **A smoothed-over disagreement.** The engines genuinely differed, the adjudicator picked
   one confidently and did not flag it.

`clip` must be one of the clips listed above and `findings` may be empty if there is honestly
nothing to report. No commentary outside the JSON.
