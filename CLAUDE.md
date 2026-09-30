# LLM Wiki Schema

## Project Structure
- `raw/` — immutable source documents. NEVER modify.
- `wiki/` — LLM-generated wiki. You own this entirely.
- `wiki/index.md` — master catalog. Update on every ingest.
- `wiki/log.md` — append-only activity log.
- `wiki/research-directions.md` — open research questions across the wiki, grouped by theme. Check and update on every ingest and every lint.

## Page Conventions
Every wiki page MUST have YAML frontmatter:
```
---
title: Page Title
type: concept | entity | source-summary | comparison | directions
sources: [list of raw/ files referenced]
related: [list of wiki pages linked]
created: YYYY-MM-DD
updated: YYYY-MM-DD
confidence: high | medium | low
---
```

## Ingest Workflow
When I say "ingest [filename]":
1. Read the source file in raw/
2. Discuss key takeaways with me 
3. Embed the figures (via markdown image links to raw/assets/) and ensure all paper tables, figures, images are reproduced in the source files
4. Discuss the main limitations
5. Create/update a summary page in wiki/sources/
6. Update wiki/index.md
7. Update all relevant concept and entity pages
8. Check wiki/research-directions.md against the paper and update it: mark
   questions it answers or partly answers, add its evidence to questions it
   bears on, and add the new questions it opens
9. Append an entry to wiki/log.md

## Research Directions
`wiki/research-directions.md` is the wiki-wide list of open research
questions, grouped by theme. It states each question briefly with the
evidence so far and links to the pages whose "Open Questions" sections carry
the detail. It does not hold experiment plans or cost estimates.

Each entry:
```
### The question, as a heading
- **Status**: open | partially answered
- **Known so far**: the evidence to date, with numbers and [[page]] citations
- **See**: [[pages]] that discuss it
- **Updated**: YYYY-MM-DD
```

Rules:
- Check and update the file on every ingest and every lint, without asking
  first. Report what changed in the discussion and in the log entry.
- Every claim must trace to a wiki page; no questions from general knowledge
  alone.
- When a paper answers a question, move the entry to the "Answered" section
  with the paper and its result in one line. Never delete an entry.
- Keep it consistent with the page-level "Open Questions": when one is
  annotated or answered, update the other.

## Query Workflow
When I ask a question:
1. Read wiki/index.md to find relevant pages
2. Read those pages
3. Synthesize an answer with [[wiki-link]] citations
4. If the answer is valuable, offer to file it as
   a new wiki page

## Lint Workflow
When I say "lint":
1. Check for contradictions between pages
2. Find orphan pages with no inbound links
3. List concepts mentioned but lacking own page
4. Check for stale claims superseded by newer sources
5. Suggest questions to investigate next
6. Check and update wiki/research-directions.md: evidence that has since
   been corrected, questions answered by later ingests but still marked
   open, and page-level Open Questions of wiki-wide interest that are
   missing from it