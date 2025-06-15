# Chapter 1: Introduction — Detailed Plan

## 1.1 Opening Context & Motivation
- Introduce recommender systems: their societal and economic impact.
- Define the central challenge: the feature engineering bottleneck.
- Motivate the need for interpretable, intelligent automation (vs. black-box AutoFE).
- Reference key foundational works (Resnick et al., Koren et al., Netflix Prize, etc.).
- Cite relevant statistics/market impact if available.

## 1.2 Evolution of Recommender Systems (Integrated Literature Review)
- Trace the development from collaborative filtering to deep learning and LLM-based approaches.
  - Early collaborative filtering (GroupLens, Resnick et al.)
  - Matrix factorization (Koren et al.)
  - Deep learning (NCF, DeepFM, BERT4Rec)
  - Knowledge graphs, hybrid models
- Highlight how advances in model complexity have not solved the feature bottleneck.
- Summarize limitations of traditional and combinatorial AutoFE (Featuretools, etc.).
- Introduce the emergence of LLMs for data science and feature engineering.
- Critically review recent LLM-based and agentic approaches (FEBP, RecMind, AlphaQuant, etc.).
- Ensure every claim is supported by a citation (from literature review or new search if missing).
- Double-check all references for correctness and BibTeX completeness.

## 1.3 Section: Large Language Models (LLMs) in Recommender Systems
- Explain what LLMs are (architecture, capabilities, recent advances).
- Discuss how LLMs are being applied to recommender systems:
  - LLMs for user/item representation
  - LLMs for feature engineering (FEBP, RAG, Chain-of-Thought, etc.)
  - LLMs for dialogue-based and conversational recommenders
  - LLM-powered agentic systems (collaborative, multi-agent setups)
- Summarize strengths and current limitations (data efficiency, explainability, etc.).
- Draw on both the planning report and up-to-date literature.
- Rigorously verify all references and cite correctly.

## 1.4 Problem Statement & Research Gap
- Articulate the gap: lack of domain-aware, collaborative, interpretable AutoFE for recommender systems.
- State why existing LLM and AutoFE systems fall short (evidence from literature).
- Explicitly define the research questions/hypotheses for AGENTIC.
- Reference planning report and literature review.

## 1.5 Project Contribution & System Overview
- Summarize the AGENTIC system and its core innovations (multi-agent, bilevel optimization, etc.).
- Briefly describe the architecture and workflow (with reference to a figure).
- Position your contribution relative to prior work (table/figure).
- Ensure all claims are cited and references are checked for accuracy.

## 1.6 Experimental Plan & Evaluation Criteria
- Outline the evaluation methodology (baselines, ablations, metrics).
- Emphasize reproducibility, logging, and rigorous comparison.
- Cite relevant evaluation standards and prior benchmarks.

## 1.7 Structure of the Thesis
- Briefly describe what each chapter will cover.

---

# Reference & Figure Plan

- For each section, list all references to be included, with a checkbox for verification.
- For each figure/table, include:
  - Title, purpose, and draft description
  - Data source or method of generation
  - Vector format requirement (SVG, PDF, TikZ)

---

# Reference Management Checklist
- Extract all references from the literature review and planning report.
- For each, verify:
  - Correct citation in text (format, placement)
  - Complete BibTeX entry
  - Consistency with thesis style
- For missing/incomplete references, search online for correct details (arXiv, publisher, DOI).
- Maintain a running list of all references, with status (checked/needs update).

# Figure/Table Management Checklist
- Draft all required figures/tables in a list, with:
  - Section, caption, data source, vector format
  - Status (drafted/needs revision/final)
- Ensure all figures are referenced in the text and have self-contained, informative captions.
- Combine related plots where possible for clarity.

---

# Self-Critique Protocol (per section)
- Is the language formal, precise, and clear?
- Is every claim/evidence supported by a correct citation?
- Are all references double-checked and BibTeX-complete?
- Does the narrative follow a logical, funnel structure?
- Are figures/tables clear, well-labeled, and described?
- Are technical terms defined and jargon avoided?
- Is the contribution and research gap unambiguously stated?
- Are supervisor’s and BICI Lab’s structural/figure guidelines followed?

---

# Next Steps
- Await user approval of this plan.
- Upon approval, begin drafting each section, performing self-critique at each stage.
- Maintain and update reference and figure checklists throughout drafting.
