# IGUANA Domain-Specific Use Cases

This directory implements four representative deployment domains. These modules demonstrate how IGUANA's out-of-band supervisor swarm and dynamic context-aware threshold adjustments function in real-world, high-stakes environments where traditional static guardrails are unsuitable.

## Use Cases

### 1. Clinical Healthcare and Mental Health Applications
* **Module**: [iguana_clinical_usecase.erl](iguana_clinical_usecase.erl)
* **Description**: Traditional guardrails often trigger categorical refusals (selective refusal bias) when processing wellness or clinical guidance queries containing sensitive keywords. This module shifts the Meta-Guard context to the strict `clinical` domain and utilizes logit soft-correction to steer the generative distribution away from clinical overreach (such as diagnostics or prescriptions) while maintaining the response's original wellness context.

### 2. Water Reuse–Food Circular Economy
* **Module**: [iguana_water_usecase.erl](iguana_water_usecase.erl)
* **Description**: Multi-stakeholder decision support in wastewater reuse involves conflicting demands (engineering constraints vs. agricultural soil salinity and contaminant limits). Instead of applying a binary block for a fertigation query, this module injects a soft correction to steer recommendations toward safe blending proportions, risk-mitigated reuse alternatives, or regulatory coordination.

### 3. Financial Services and Algorithmic Risk Assessment
* **Module**: [iguana_financial_usecase.erl](iguana_financial_usecase.erl)
* **Description**: Autoregressive models trained on historical financial data often exhibit geographic biases (e.g., deprioritizing loans for small businesses in rural/underrepresented areas). This module shifts context to `financial` and applies parallel risk-assessment correction vectors to balance recommendation logits across regions.

### 4. Legal and Governmental Decision Support
* **Module**: [iguana_legal_usecase.erl](iguana_legal_usecase.erl)
* **Description**: Automated summaries or sentencing rationales within legal decision-support systems risk replicating historical socioeconomic and demographic biases. Rather than enforcing blocklists that prevent thorough case analysis, this module detects bias markers and mathematically rebalances generation probabilities to ensure fair, context-preserving summaries.

---

## Verification

The correctness of these use-case logic flows is verified via the `iguana_usecases_SUITE` common test suite located at `test/iguana_usecases_SUITE.erl`.
