---
name: paper-layered-explainer
description: >-
  Use this skill when the user uploads or references a long, dense paper and wants it rewritten in full, sequentially, in a technical register and separately in a plain-language register — two complete parallel walkthroughs, not a single-altitude summary and not an interleaved paragraph-by-paragraph comparison. Trigger on phrasing like "rewrite this paper simply, all the way through," "give me a full technical version and a full ELI5 version," "walk through this paper twice, once for experts and once for beginners," or any request for two complete sequential renderings of a document at different reading levels. Distinct from paper-deep-dive, which produces a research-analyst report for AI/ML papers and skips to synthesis; this skill walks the paper in its own order twice — once fully technical, once fully plain-language — for any long paper in any field. Always run the three-pass read below first; don't summarize from a single linear read.
---

# Paper Layered Explainer

Turns a long paper into two complete, sequential rewrites — a full technical walkthrough and a full plain-language walkthrough, each running start to finish in the paper's own order — so a reader can read either one straight through as its own document, and cross-reference the other by paragraph number if they want to.

## The actual failure mode this skill has to avoid

If both walkthroughs faithfully render *every* paragraph, including funding acknowledgments, citation-dump related-work paragraphs, and repeated notation reminders, you get two long documents instead of one, with the added length buying nothing. The filtering step below (deciding what's substantive) isn't optional busywork — it's what keeps both walkthroughs worth reading start to finish rather than a slog.

The other failure mode: manufacturing a plain-language gloss for a paragraph that's pure algebra with no conceptual payload. A fake analogy bolted onto a derivation sounds like an explanation but conveys nothing, and is worse than admitting there's nothing to simplify.

**The tradeoff worth naming:** splitting into two full sequential documents (rather than interleaving both versions under each paragraph) means a reader who wants to compare "how was this one specific paragraph rendered at each level" has to flip between two sections instead of reading one block. The fix is consistent numbered anchors — every substantive paragraph gets the same `[P#]` tag in both walkthroughs, so cross-referencing is a search away even though the two versions aren't sitting next to each other.

## Pass 1: Read the whole paper, once, for the arc

Before annotating anything, read start to finish. Build (for your own use, not for the output) a sense of: the paper's overall argument, the handful of key terms and where they're first defined, and what job each section does in the larger argument. This matters because paragraph-level explanations written in isolation — without knowing what comes before or after — tend to re-explain the same term five different ways or contradict themselves across sections. Pass 1 is what keeps Pass 3 consistent.

If the source is a PDF, use the pdf-reading skill. For anything with a figure, table, or equation, rasterize and actually look at the page — a paragraph that references "as shown in Figure 3" can't be explained at any register without seeing Figure 3.

## Pass 2: Re-read section by section

Go back through one section at a time. For each section, decide:

- **Its role in the paper's argument** (motivation, method, result, discussion, limitation, etc.) — one line.
- **Which paragraphs are substantive vs. boilerplate.** Substantive = introduces a claim, defines a term, describes a method step, presents or interprets a result, states an assumption or a limitation. Boilerplate = acknowledgments, funding/conflict-of-interest statements, author contribution statements, reference lists, appendix legalese, or a paragraph that's purely a list of citations with no argument of its own. When in doubt, keep it — the coverage notes at the end are where you account for anything left out, so exclusion should be visible, never silent.
- **Which paragraphs lean on a specific figure, table, or equation** — flag these so Pass 3 doesn't paraphrase blind.
- **Appendices, specifically:** don't default to full paragraph-level treatment. Most appendix content (hyperparameter tables, tuning ranges, implementation-detail lists) is bookkeeping rather than argument, and belongs in coverage notes, not a translated walkthrough. The exception is an appendix paragraph that defines something genuinely new and load-bearing for a result already covered in the main text (e.g., the exact function used to implement a constraint mentioned but not spelled out in Section 3 or 5) — those earn a `[P#]` entry; everything else gets summarized in coverage notes instead.

Once the substantive list for a section is settled, assign each one a sequential `[P#]` tag continuing across the whole paper (not restarting per section). This tag is what keeps the two walkthroughs in Pass 3 aligned to each other even though they're not physically adjacent.

**Size guardrail:** don't check this per section — a paper can pass a 15-per-section check easily while still totaling 40-50+ substantive paragraphs once every small section is added up, which produces the same bloated-document problem the check exists to prevent. After finishing the section-by-section pass, add up the `[P#]` count across the *whole* main text. If it's past ~35–40, stop before Pass 3 and ask the user whether they want full paragraph-level coverage or a condensed version that groups tightly-related paragraphs into single passages. Don't silently truncate or silently expand — this is a real tradeoff (completeness vs. a document someone will actually finish reading) and it's the user's call, not a default to guess at.

## Pass 3: Two full sequential walkthroughs

Don't interleave registers under each paragraph. Instead, do two complete sweeps through the same ordered list of `[P#]`-tagged paragraphs from Pass 2:

**Sweep 1 — Technical walkthrough.** Render every substantive paragraph, in paper order, in the paper's own register: real terminology, real notation, no hand-holding. If there's an equation, keep it, and gloss what each symbol means in one clause. Read start to finish, this should feel like a faithful, slightly-more-explicit rewrite of the paper — not a summary.

**Sweep 2 — Plain-language walkthrough.** Render the same `[P#]`-tagged paragraphs, same order, for a smart non-specialist: analogy or everyday comparison, no jargon, no notation. If a simplification drops something that matters to a specialist, say so in a half-sentence rather than pretending nothing was lost. For a paragraph that's pure derivation or notation with no conceptual content beyond "algebra happens here," say exactly that instead of inventing an analogy that isn't really about anything.

Write each sweep as its own complete pass rather than generating both registers for `[P1]` before moving to `[P2]` — the point of doing it this way is that each walkthrough reads as a coherent standalone document, not a set of paired fragments.

## Output: the report

Save as `<slug>-layered-report.md`, where slug is first author + year (e.g. `vaswani2017-attention-layered-report.md`), consistent with paper-deep-dive's naming so the two can sit side by side if both are run on the same paper.

```
# [Paper title]
## TL;DR
(3-4 sentences, plain language: what this paper is about and why someone would read it)
## Reading map
(one short paragraph: the paper's argument arc, and the handful of key terms introduced early — so the plain-language walkthrough below doesn't have to re-explain a term from scratch every time it recurs)

## Technical Walkthrough
### [Section name]
**Role in the paper:** ...
[P1] ...
[P2] ...
(repeat per section, in the paper's own order, continuing the [P#] numbering across the whole paper)

## Plain-Language Walkthrough
### [Section name]
[P1] ...
[P2] ...
(same paragraphs, same [P#] tags, same order — so a reader can jump to [P7] in either walkthrough and know it's the same underlying passage)

## Coverage notes
(what was excluded as boilerplate, what was condensed into a single passage, and why — this is what makes the filtering in Pass 2 auditable rather than silent)
## Reference
(citation, link if available)
```

## Relationship to paper-deep-dive

This skill is a reading companion, not a critical analysis — it doesn't judge what's novel, whether the evaluation is fair, or produce an architecture diagram. If the user wants that too, paper-deep-dive covers it. The two are complementary and can both be run on the same paper, but don't merge them into one file — they answer different questions ("what does this paragraph actually mean, at my level" vs. "is this paper any good").