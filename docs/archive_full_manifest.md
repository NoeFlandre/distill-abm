# Archive Full Manifest

- Total files: 592
- Classification counts:
  - `experiment_setting`: 38
  - `historical_nonruntime`: 332
  - `human_ground_truth`: 32
  - `legacy_visualization`: 184
  - `prompt_reference`: 6
- Action counts:
  - `archive_separately`: 184
  - `discard_with_rationale`: 39
  - `migrate`: 76
  - `retain_record_only`: 293
- Unresolved mappings: 0

## Sample Rows

| path | classification | action | target_path |
| --- | --- | --- | --- |
| `archive/.DS_Store` | `historical_nonruntime` | `retain_record_only` | `-` |
| `archive/legacy_repo/.DS_Store` | `historical_nonruntime` | `retain_record_only` | `-` |
| `archive/legacy_repo/Code/.DS_Store` | `historical_nonruntime` | `retain_record_only` | `-` |
| `archive/legacy_repo/Code/Evaluation/.DS_Store` | `historical_nonruntime` | `retain_record_only` | `-` |
| `archive/legacy_repo/Code/Evaluation/DOE/.DS_Store` | `historical_nonruntime` | `retain_record_only` | `-` |
| `archive/legacy_repo/Code/Evaluation/DOE/Archives/1 and -1 Sorted.csv` | `historical_nonruntime` | `retain_record_only` | `-` |
| `archive/legacy_repo/Code/Evaluation/DOE/Archives/Cédric Version/DoEFeatureImportance.ipynb` | `historical_nonruntime` | `discard_with_rationale` | `-` |
| `archive/legacy_repo/Code/Evaluation/DOE/Archives/Cédric Version/FinalResultsYesNo.csv` | `experiment_setting` | `migrate` | `tests/fixtures/notebook_parity/experiment_settings/Evaluation/DOE/Archives/Cédric Version/FinalResultsYesNo.csv` |
| `archive/legacy_repo/Code/Evaluation/DOE/Archives/Cédric Version/factorial_contributions.csv` | `historical_nonruntime` | `retain_record_only` | `-` |
| `archive/legacy_repo/Code/Evaluation/DOE/Archives/Cédric DoE Analysis averaging the repetitions.ipynb` | `historical_nonruntime` | `discard_with_rationale` | `-` |
| `archive/legacy_repo/Code/Evaluation/DOE/Archives/test2.csv` | `historical_nonruntime` | `retain_record_only` | `-` |
| `archive/legacy_repo/Code/Evaluation/DOE/DoE.ipynb` | `historical_nonruntime` | `discard_with_rationale` | `-` |
| `archive/legacy_repo/Code/Evaluation/DOE/FinalResultsYesNo.csv` | `experiment_setting` | `migrate` | `tests/fixtures/notebook_parity/experiment_settings/Evaluation/DOE/FinalResultsYesNo.csv` |
| `archive/legacy_repo/Code/Evaluation/DOE/anova_factorial_contributions.csv` | `historical_nonruntime` | `retain_record_only` | `-` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/ Training/PresentationRaters.pptx` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/ Training/PresentationRaters.pptx` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/ Training/PresentationRaters2.pptx` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/ Training/PresentationRaters2.pptx` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/ Training/UnrelatedCase.xlsx` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/ Training/UnrelatedCase.xlsx` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/All Results/AllResultsAssessedSoFar.numbers` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/All Results/AllResultsAssessedSoFar.numbers` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/All Results/V2FaunaGrazingButRoleMessedUp.csv` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/All Results/V2FaunaGrazingButRoleMessedUp.csv` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/All Results/V2Milk.csv` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/All Results/V2Milk.csv` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Interrater Agreement/Training 1/Evaluation.xlsx` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Interrater Agreement/Training 1/Evaluation.xlsx` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Interrater Agreement/Training 1/Interrater Agreement.xlsx` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Interrater Agreement/Training 1/Interrater Agreement.xlsx` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Interrater Agreement/Training 1/RatersSheetTemplate1.xlsx` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Interrater Agreement/Training 1/RatersSheetTemplate1.xlsx` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Interrater Agreement/Training 2/Interrater Agreement copy.xlsx` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Interrater Agreement/Training 2/Interrater Agreement copy.xlsx` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Interrater Agreement/Training 2/Rating2 - EVALUATED.xlsx` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Interrater Agreement/Training 2/Rating2 - EVALUATED.xlsx` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Interrater Agreement/Training 2/Rating2.xlsx` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Interrater Agreement/Training 2/Rating2.xlsx` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Interrater Agreement/Training 3/Cedric /RatersSheetTemplateImagesOverCell(1).xlsx` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Interrater Agreement/Training 3/Cedric /RatersSheetTemplateImagesOverCell(1).xlsx` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Interrater Agreement/Training 3/Interrater Agreement copy.xlsx` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Interrater Agreement/Training 3/Interrater Agreement copy.xlsx` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Interrater Agreement/Training 3/Siva/Ratings3.xlsx` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Interrater Agreement/Training 3/Siva/Ratings3.xlsx` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Others/AllResultsAssessedSoFar.xlsx` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Others/AllResultsAssessedSoFar.xlsx` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Others/QualitativeAssessment.xlsx` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Others/QualitativeAssessment.xlsx` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Others/Training.xlsx` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Others/Training.xlsx` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Rating 1/RatersSheetTemplate.xlsx` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Rating 1/RatersSheetTemplate.xlsx` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Rating 1/RatersSheetTemplate2.xlsx` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Rating 1/RatersSheetTemplate2.xlsx` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Rating 1/RatersSheetTemplateImagesOverCell.xlsx` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Rating 1/RatersSheetTemplateImagesOverCell.xlsx` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Rating 2/RatersSheetTemplateImagesOverCell.xlsx` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Rating 2/RatersSheetTemplateImagesOverCell.xlsx` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Rating 3/Cedric/RatersSheetTemplateImagesOverCell(1).xlsx` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Rating 3/Cedric/RatersSheetTemplateImagesOverCell(1).xlsx` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Rating 3/RatersSheetTemplateImagesOverCell.xlsx` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Rating 3/RatersSheetTemplateImagesOverCell.xlsx` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Rating 3/Siva/Ratings3.xlsx` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Rating 3/Siva/Ratings3.xlsx` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Rubrics/2023 BEA 1.32.pdf` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Rubrics/2023 BEA 1.32.pdf` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Rubrics/2307.08161v1.pdf` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Rubrics/2307.08161v1.pdf` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Rubrics/2310.08433v1.pdf` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Rubrics/2310.08433v1.pdf` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Rubrics/Rubrics.pdf` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Rubrics/Rubrics.pdf` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Rubrics/Rubrics.pptx` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Rubrics/Rubrics.pptx` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Rubrics/Table main.tex` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Rubrics/Table main.tex` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/Rubrics/Tian EDM LLM 2024.pdf` | `human_ground_truth` | `migrate` | `tests/fixtures/notebook_parity/human_reference/Rubrics/Tian EDM LLM 2024.pdf` |
| `archive/legacy_repo/Code/Evaluation/Human Assessment/~$PresentationRaters.pptx` | `historical_nonruntime` | `retain_record_only` | `-` |
| `archive/legacy_repo/Code/Evaluation/Making the Sheet Structure/.ipynb_checkpoints/Final Sheet-checkpoint.ipynb` | `historical_nonruntime` | `discard_with_rationale` | `-` |
| `archive/legacy_repo/Code/Evaluation/Making the Sheet Structure/Archives/Final Sheet/updated_structured_data.csv` | `historical_nonruntime` | `retain_record_only` | `-` |
| `archive/legacy_repo/Code/Evaluation/Making the Sheet Structure/Archives/Final Sheet/updated_structured_data2.csv` | `historical_nonruntime` | `retain_record_only` | `-` |
