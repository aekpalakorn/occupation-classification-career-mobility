# Leveraging Large Language Models for Career Mobility Analysis

## Overview

This repository contains the source code for the data analysis and framework, including the FewSOC prompting framework for O*NET-SOC classification, used in the paper:

**"Leveraging Large Language Models for Career Mobility Analysis: A Study of Gender, Race, and Job Change Using U.S. Online Resume Profiles."**

This code base supports replication and extension of the quantitative analyses presented in the paper. It focuses on demonstrating the LLM-based occupational classification pipeline and subsequent statistical analysis of early-career upward mobility outcomes for college-educated workers.

**Important Note For Reproducibility**

* The code base was developed using a proprietary, licensed commercial dataset (Lightcast data). The raw data files are NOT included in this repository and cannot be publicly shared.
* All file names used in the code are placeholders. Users must substitute them with their own files.

## Repository Structure

The project code is organized into two primary directories:

| **Directory** | **Content**                       | **Description**                                                                                                                                                                                                                   |
| ------------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/notebooks/` | Jupyter Notebook files (`.ipynb`) | Contains notebooks for dataset preprocessing and construction, regression analyses, multilevel sensitivity analysis, and figure generation.                                                                                       |
| `/fewsoc/`    | Core Python modules (`.py`)       | Contains runnable Python files for the FewSOC prompting framework.                                                                                                                                                                |
| `/data/`      | CSV / TXT files                   | Contains the FewSOC prompt template file, aggregate datasets used for dataset construction, and files required for visualization. Original proprietary Lightcast data is not included. |

## FewSOC Prompting Framework

The FewSOC framework is implemented in `/fewsoc/`. It achieves improved occupational classification by leveraging LLMs with a batched few-shot prompting approach to infer O*NET-SOC titles and codes from job titles and company names.

## 1. Generate Predictions

The first step is to classify job titles into O*NET-SOC codes using `classify_soc.py`.

**Input CSV Requirements:**

The input CSV must have at least **two columns**:

| Column Name | Description                                                                              |
| ----------- | ---------------------------------------------------------------------------------------- |
| `task_id`   | Unique identifier for each row/job title. Used to track API responses and match outputs. |
| `sentence`  | Job title or job description text to be classified into O*NET-SOC codes.                 |

**Notes:**

* Each row represents a single "job title, company name" for which you want SOC code predictions.
* Additional columns are allowed but will be ignored by the script.
* `task_id` must be unique, as it is used to merge predictions back into the original dataset.

**Example Input CSV:**

| task_id | sentence            |
| ------- | ------------------- |
| 1       | Software Engineer, Company A   |
| 2       | Marketing Manager, Company B   |
| 3       | Data Scientist, Company C |

**Example Command:**

```bash
python classify_soc.py \
    --model gpt-3.5-turbo \
    --temperature 0 \
    --prompt_file ../data/onet_prediction_prompt_template_v1.25.txt \
    --input_csv ../data/job_titles.csv \
    --output_csv ../data/predictions.csv \
    --raw_output_json ../data/api_responses.json \
    --log_file ../logs/gpt-3.5-turbo.log
```

**Output:**

* `predictions.csv` — Contains raw SOC predictions with at least `pred_soc_title` and `pred_soc_code` columns.
* `api_responses.json` — Stores raw API responses.
* `log_file` — Tracks progress, warnings, and errors.

---

## 2. Post-process SOC Titles

The second step maps the raw predicted SOC titles to canonical O*NET titles using `postprocess_soc_titles.py`.

**Input CSV Requirements:**

1. **Predictions CSV (`pred_csv`)**
   This is the output of `classify_soc.py` and must contain at least:

| Column Name      | Description                                                            |
| ---------------- | ---------------------------------------------------------------------- |
| `pred_soc_title` | SOC title predicted by the LLM/classifier.                             |
| `pred_soc_code`  | SOC code predicted (may be empty or inaccurate before postprocessing). |

**Notes:**

* Each row corresponds to a single job title or position.
* Additional columns (e.g., `task_id`, `sentence`) are allowed but ignored by the script.
* The script corrects hallucinated or slightly mismatched titles by mapping them to the closest canonical O*NET title.

2. **Canonical O*NET SOC CSV (`onet_csv`)**

| Column Name | Description             |
| ----------- | ----------------------- |
| `title`     | Official SOC job title. |
| `code`      | Corresponding SOC code. |

**Notes:**

* Any empty or missing titles are ignored.
* The script uses tokenized similarity and deterministic tie-breaking to map predictions to canonical titles.

**Example Command:**

```bash
python postprocess_soc_titles.py \
    --onet_csv ../data/onet-soc_2019.csv \
    --pred_csv ../data/predictions.csv \
    --output_csv ../data/predictions_postprocessed.csv
```

**Output:**

* `predictions_postprocessed.csv` — Contains the same columns as the predictions CSV, with `pred_soc_title` and `pred_soc_code` corrected to match canonical O*NET titles.
* Console logs report how many hallucinated titles were fixed and how many could not be mapped.

## 3. Compute Accuracy Scores

The final step evaluates the correctness of predicted SOC labels against ground truth annotations using `compute_accuracy.py`.

**Input CSV Requirements:**

1. **Predictions CSV (`data_csv`)**
   This is the post-processed predictions CSV (output of `postprocess_soc_titles.py`) and must contain at least:

| Column Name      | Description                                                                 |
| ---------------- | --------------------------------------------------------------------------- |
| `pred_soc_title` | SOC title predicted (after postprocessing).                                 |
| `pred_soc_code`  | SOC code predicted (after postprocessing).                                  |
| `sentence`       | Original job title or job description. Used for matching with ground truth. |

2. **Ground Truth CSV (`answer_csv`)**
   Contains manually annotated SOC labels for evaluation and must include:

| Column Name             | Description                                                                                 |
| ----------------------- | ------------------------------------------------------------------------------------------- |
| `<answer_label_column>` | Semicolon-separated list of ground truth SOC labels for each sentence.                      |
| `sentence`              | Original job title or job description. Must match the `sentence` column in predictions CSV. |

**Notes:**

* The script matches predicted labels against the set of ground-truth labels using an any match criterion.
* Both CSVs must have a `sentence` column for merging predictions with ground truth.
* The ground-truth column may contain multiple SOC labels per row, separated by semicolons (`;`).

**Example Command:**

```bash
python compute_score.py \
    --data_csv ../data/predictions_postprocessed.csv \
    --answer_csv ../data/tgre_zeroshot_gpt4_answers.csv \
    --answer_label_column answer \
    --output_csv ../data/prediction_comparison.csv
```

**Output:**

* `prediction_comparison.csv` — Contains merged data with predicted and ground-truth labels, cleaned labels, and a `match` column indicating whether the prediction matched any ground truth label.
* Console output reports:

  * Total number of samples compared
  * Number of correctly matched samples
  * Accuracy score (Any Match)
``

## Jupyter Notebooks for Upward Mobility Analysis

There are four main notebooks:

* **Dataset Construction.ipynb**: Constructs a cleaned, enriched career trajectories dataset of Bachelor’s degree holders, combining job and education records with demographic predictions and state-level economic data. Optimized for regression analyses of early-career mobility.

* **Main Analysis.ipynb**: Fits logistic regression models to assess how gender, race, and job change types predict upward mobility. Explores intersectional disparities via interactions and gender-stratified models.

* **Sensitivity Analysis.ipynb**: Performs multilevel Bayesian logistic regressions to evaluate robustness of main fixed-effect estimates under hierarchical random effects for occupation, industry, state, firm size, and career entry cohort.

* **Figures.ipynb**: Generates all figures from the paper, visualizing gender, race, and job change effects in early-career mobility. Reproduces the key plots for analysis and presentation purposes.

## Files Description

The `/data/` folder contains the following files used in analysis, visualization, and FewSOC classification:

| File Name                                   | Description                                                                                                                        |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `1998_2022_real_gdp_by_state.csv`           | State-level real GDP from 1998 to 2022 (source: BLS)                                                                               |
| `major_occupation_transitions_y1-5.csv`     | Early career (years 1–5) job transitions between major occupations, aggregated from Career229K dataset                             |
| `occupation_growth_y1-5.csv`                | Early career (years 1–5) growth rates of major occupations, aggregated from Career229K dataset                                     |
| `onet-soc_2019.csv`                         | O*NET-SOC 2019 taxonomy                                                                                                            |
| `onet_prediction_prompt_template_v1.25.txt` | FewSOC prompt template for LLM-based SOC classification                                                                            |
| `sensitivity_coefficients.csv`              | Coefficients for Models 1–4 and their multilevel sensitivity counterparts                                                          |
| `table4_coefficients.csv`                   | Coefficients for Models 1–4 as reported in Table 4 of the paper                                                                    |
| `tgre_zeroshot_gpt4_answers.csv`            | GPT-4o annotated ground-truth SOC titles and codes for occupation classification evaluation                                        |
| `wage_interpolated_1999_2022_soc_2019.csv`  | State-level wage data from 1999–2022 (source: BLS). SOC codes mapped to O*NET-SOC 2019 via crosswalk files; missing values imputed |
| `soc_non2019_to_2019_mapping.csv`  | Mapping of legacy 8-digit SOC codes to their corresponding O*NET-SOC 2019 8-digit codes |


## Setup and Installation

### Prerequisites

* Python 3.11+
* Required Python packages:

```
arcplot==1.0.0
matplotlib==3.8.2
numpy==1.23.5
openai==0.28.1
pandas==2.3.3
protobuf==6.32.1
requests==2.32.5
seaborn==0.13.2
statsmodels==0.14.0
tabulate==0.8.10
tenacity==9.1.2
```

### Installation Steps

1. **Clone the repository:**

```bash
git clone https://github.com/aekpalakorn/occupation-classification-career-mobility.git
cd <your-local-directory>
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Prepare your datasets:**

* Place your pre-processed data files in the `data/` directory.
* **Note:** Original Lightcast data cannot be shared publicly; file names in notebooks are placeholders.

4. **Run analyses:**

* Open notebooks in Jupyter or JupyterLab and execute cells sequentially.
* Results (regression tables, sensitivity analyses, figures) will be saved in `results/` or `data/` folders.
