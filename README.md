# ORACLEAI

> *"What's really going to bake your noodle later on is, would you still have broken it if I hadn't said anything?"*
> — **The Oracle**

---

**The Self-Fulfilling Prophecy Problem in AI Prediction Systems**

`PROJECT STATUS: RESEARCH PHASE`

## Overview

The Oracle's predictions change the future by being known. AI prediction systems face the same paradox: when an AI predicts someone will default on a loan, and they're denied the loan because of the prediction, the prediction becomes unfalsifiable.

This is one of the deepest problems in deployed AI: **performative prediction**. AI systems that predict human behavior change that behavior by existing. This project builds a formal framework for detecting and correcting performative prediction loops.

## Research Question

Credit scores change lending behavior. Recidivism predictions change sentencing. Recommendation algorithms change preferences. How do we solve the Oracle's Paradox in AI?

## Methodology

### 1. Formal Framework
- Define performative prediction: `P(outcome | prediction known) != P(outcome | prediction unknown)`
- Model the feedback loop: prediction -> action -> outcome -> validates prediction
- Identify conditions under which the loop is stable vs unstable

### 2. Empirical Tests
- **Credit scoring:** simulate loan decisions with/without AI predictions
- **Content recommendation:** measure how recommendations change preferences
- **Predictive policing:** model the feedback loop of prediction -> surveillance -> detection

### 3. Oracle-Corrected Methods
- **Counterfactual predictions:** what would happen WITHOUT the prediction?
- **Causal predictions:** separate correlation from causation in feedback loops
- **Self-aware predictions:** systems that model their own influence on outcomes

### 4. The Vedantic Connection
**Karma** — actions (predictions) create consequences that reinforce the action. The Oracle's paradox is the computational instantiation of karmic loops.

## Expected Outputs

- **Paper:** *"The Oracle's Paradox: Detecting and Correcting Self-Fulfilling Prophecies in AI Prediction Systems"*
- **Library:** `oracleai` — Python library for performative prediction correction
- **Impact:** Direct real-world impact on AI fairness and deployed systems

## Tech Stack

- Python 3.11+
- Causal inference libraries (DoWhy, CausalML)
- Simulation frameworks
- Statistical modeling (statsmodels, scipy)

---

*Part of the [Matrix Research Series](https://github.com/MukundaKatta) by [Officethree Technologies](https://github.com/MukundaKatta/Office3)*

**Mukunda Katta** · Officethree Technologies · 2026

> *"We can never see past the choices we don't understand."*
