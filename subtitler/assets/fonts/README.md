# Bundled fonts

`NotoSans-Regular.ttf` and `NotoSans-Bold.ttf`, licensed under the SIL Open Font License
1.1 (https://openfontlicense.org). Source: https://github.com/notofonts/latin-greek-cyrillic

## Why these ship inside the package

There is no font installed by default on both macOS and stock Ubuntu. macOS has Arial and
Helvetica but not DejaVu; Ubuntu has DejaVu and Liberation but not Arial. Naming a system
font in the subtitle style therefore renders differently on different machines: different
glyph metrics mean different line widths, different overflow, and a subtitle file that
looks correct on the machine that made it and wrong on the machine that plays it.

The earlier version of this pipeline used `FontName=Arial`, which silently falls back on
Ubuntu. Bundling removes the whole class of problem and makes rendered output comparable
across platforms, which the benchmark's pixel-diff test depends on.

Noto Sans covers Serbian Latin (č ć đ š ž) and Cyrillic.
