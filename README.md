# Antibody Developability Benchmark

To anyone reading this: Feel free to change the layout of this repo as much as you want!

At the moment a minimal setup:
- src/models contains example baselines: linear regression model trained on TAP descriptors, and baselines run with Tamarind.bio
- src/features contains some useful precomputed features (e.g. DeepSP structural descriptors)
- src/evaluate.py contains the evaluation code to get metrics
- data/ contains the GDPa1 dataset
- predictions/ contains the prediction CSVs for each model. Each file must have an antibody_name column, and a column named by property (e.g. HIC, Tm2, etc.)
- results/ contains the metrics for each model

The predictions/ directory is in the following format:
`dataset/model_dir/model_name.csv`
e.g. `GDPa1_cross_validation/TAP/TAP - linear regression.csv`
The directory is there for when you have multiple features from the same model.

TODOs:
- Add tutorial notebook to the repo in this format and link at the top of the README
- Add other features (structure files)
- Check results still work with the new structure
- Maybe concatenate all features together for a proper feature store? It doesn't matter that they came in different batches.
