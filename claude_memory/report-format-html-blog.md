---
name: report-format-html-blog
description: "Ryan wants final reports as an easy-to-read HTML blog post carried mainly by diagrams and graphs, not long text"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 80655dc5-1eae-4992-a93b-f6f87e8ab1d8
  modified: 2026-10-15T16:14:39.598Z
---

Final reports for Ryan should be HTML blog posts that are easy to understand, with most points expressed as
figures: drawn diagrams (pipelines, data flow, reduction trees) and graphs (bars, curves, heatmaps) rather than
paragraphs and tables of numbers.

**Why:** Ryan said (2026-10-15, about the Muse-Glimmer POC report) that he now does everything in HTML blog form
and wants what I want to say shown as pictures and graphs wherever possible.

**How to apply:** for any end-of-task report, build an HTML page (publish it as an Artifact; also commit it if the
repo keeps blogs under docs/blog), lead each section with a figure, keep prose short, and put raw numbers in
collapsible details. Related: [[museglimmer-poc-state]].
