# DAQathon Session 1 Agenda

This is the suggested agenda for the first DAQathon session. It follows the structure of the two notebooks in this repo, with the first notebook doing the main guided walkthrough and the second notebook opening the door to deeper experiments.

## Session Goals

By the end of Session 1, you should be able to:

- move from raw ONC scalar data to a format that is easier to use in ML experiments,
- inspect QC flags in context instead of treating them as abstract labels,
- train and interpret a strong tabular baseline,
- compare supervised and unsupervised workflows on the same dataset,
- and understand what sequence models like CNNs and transformers are trying to learn.

## Before The Session

Before you start, make sure you can:

1. SSH into FIR:

   ```bash
   ssh <username>@fir.alliancecan.ca
   ```

2. Clone the repo into your `HOME` space:

   ```bash
   cd "$HOME"
   git clone https://github.com/Spiffical/daqathon.git daqathon
   cd daqathon
   ```

3. Install the shared kernel once:

   ```bash
   jupyter kernelspec install --user /project/def-kmoran/shared/daqathon/kernels/daqathon-ml
   ```

4. Launch JupyterHub:

   [https://jupyterhub.fir.alliancecan.ca/](https://jupyterhub.fir.alliancecan.ca/)

5. Open the notebook from `~/daqathon/notebooks/` and choose the `Daqathon ML` kernel.

## Suggested Flow

This is a practical, discussion-friendly flow for the first session. The exact timing can flex depending on how much live experimentation the group wants to do.

### 1. Setup And Orientation

Estimated time: `10–15 min`

- confirm everyone can open the repo and select the kernel,
- explain where the shared data lives on FIR,
- explain where notebook outputs go,
- and quickly orient everyone to the structure of the main notebook.

Primary notebook section:

- `Part 1 — Orientation and Dataset`

### 2. Look At The Raw Data First

Estimated time: `15–20 min`

- inspect real rows from the raw CSVs,
- look at QC flags in context,
- talk about what one "example" means in this dataset,
- and start forming hypotheses about what might be easy or hard for a model.

Primary notebook sections:

- `Part 1 — Orientation and Dataset`

Questions to emphasize:

- What do the QC flags look like in context?
- Are the interesting events local spikes, longer drifts, or both?
- What would you want a model to pay attention to?

### 3. Raw CSV To Model-Ready Data

Estimated time: `20–25 min`

- explain why the raw ONC files are not ideal for repeated ML work,
- walk through the prep script,
- show how timestamps, measurement columns, QC flags, and metadata are cleaned and typed,
- and explain why parquet is useful here.

Primary notebook sections:

- `Part 2 — Data Preparation and Caching`

Questions to emphasize:

- What should one training example be?
- Which columns are measurements, labels, identifiers, or metadata?
- What file format makes repeated experiments faster and safer?

### 4. Build A Baseline With Random Forest

Estimated time: `25–30 min`

- turn row-level data into a tabular supervised problem,
- train a baseline Random Forest,
- inspect feature importance,
- and look at mistakes on a held-out time range.

Primary notebook sections:

- `Part 3 — Feature Engineering and Data Loading`
- `Part 4 — Random Forest`

Questions to emphasize:

- Which engineered features helped the most?
- Are mistakes driven by class imbalance, poor features, or target ambiguity?
- How different are the train, validation, and test label mixes?

### 5. Compare With Unsupervised Learning

Estimated time: `15–20 min`

- run k-means on row-level or window-level features,
- inspect clusters in feature space,
- and compare cluster assignments to the real QC flags.

Primary notebook sections:

- `Part 5 — k-means`

Questions to emphasize:

- Do the clusters correspond to meaningful operating regimes?
- Do clusters line up with QC issues?
- What can unsupervised learning tell you before you have a strong label model?

### 6. Introduce Sequence Models

Estimated time: `30–40 min`

- show how CNNs see local temporal patterns,
- show how transformers use self-attention across a whole window,
- compare `window` output mode with `per_timestep` output mode,
- and discuss what information sequence models see directly versus what the Random Forest needed us to engineer.

Primary notebook sections:

- `Part 6 — 1D CNN`
- `Part 7 — Transformer`

Questions to emphasize:

- When is one label per window too coarse?
- When does per-timestep prediction make more sense?
- What kinds of context might a transformer capture better than a CNN?

### 7. Wrap Up And Set Up Session 2

Estimated time: `10–15 min`

- review what changed most when the input representation changed,
- review how target choice affects model behavior,
- and point toward the advanced notebook for search, feature engineering, and stronger sequence models.

Primary notebook sections:

- `Part 8 — Reflection and Next Steps`
- `notebooks/advanced_session1_qc_workflow.ipynb`

## How The Two Notebooks Fit Together

### Main notebook

Use [intro_session1_qc_workflow.ipynb](/home/sbialek/ONC/DAQathon/notebooks/intro_session1_qc_workflow.ipynb) for the first session:

- explore the data,
- understand the prep workflow,
- build the first baseline models,
- and leave with good questions.

### Advanced notebook

Use [advanced_session1_qc_workflow.ipynb](/home/sbialek/ONC/DAQathon/notebooks/advanced_session1_qc_workflow.ipynb) when you want to go deeper:

- hyperparameter search,
- richer feature engineering,
- stronger tree models,
- target redesign,
- and sequence-labeling experiments.

## If You Have Extra Time

Good live experiments for the end of Session 1:

- switch `DATA_FRACTION` upward and compare stability,
- change the dataset profile,
- compare `window` versus `per_timestep` for the sequence models,
- inspect how the split distributions change across datasets,
- and try a different date range in the model demos.

## Main Takeaway

The biggest lesson from Session 1 is that model choice is only part of the problem. How you define the target, prepare the data, choose the input representation, and split the data in time often matters just as much as the model itself.
