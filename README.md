Machine Failure Predictor

Predicts whether a factory machine will fail, using its air temperature, process temperature, rotational speed, torque, tool wear and product type. Built as a Machine Learning Lab mini project (B.Tech CSE).

Live app: https://machine-failure-prediction-by4kuv5xzqhgpuw4zakfw6.streamlit.app/

Team: Bhagyashree Umap, Radhika Asati, Aarya Pachkhede 

<img width="1898" height="975" alt="Screenshot 2026-10-26 172839" src="https://github.com/user-attachments/assets/2a2872c3-c23b-476a-9cca-dce7ecc7c6da" />

Why this project?

Machines that fail without warning cause downtime and repair costs. A model that warns operators early lets them repair the machine at a planned time.

The difficulty is that failures are rare. In the AI4I 2020 dataset only 339 of 10,000 rows (3.39%) are failures, so a model that always says "no failure" is right 96.6% of the time while catching nothing. In our notebook this dummy model scores 0.966 accuracy and an F1 of 0.000 for failures. Because of this, models are judged on precision, recall, F1 and PR-AUC for the failure class, not on accuracy.

Dataset

AI4I 2020 Predictive Maintenance Dataset from the UCI Machine Learning Repository (also on Kaggle). It has 10,000 rows and 14 columns and is a synthetic dataset made to look like real milling machines.

Column	Use
Type (L, M, H)	Feature, one-hot encoded
Air temperature [K], Process temperature [K]	Features
Rotational speed [rpm], Torque [Nm], Tool wear [min]	Features
Machine failure	Target (1 = failed, 0 = fine)
UDI, Product ID	Dropped (only labels)
TWF, HDF, PWF, OSF, RNF	Dropped (they give the reason for the failure, so they leak the answer)

Failures by cause (a machine can have more than one): tool wear (TWF) 46, heat dissipation (HDF) 115, power (PWF) 95, overstrain (OSF) 98, random (RNF) 19.

Approach
EDA: class balance, histograms, box plots, correlation, failure rate by type, torque against speed.
Cleaning: no missing values, no duplicate rows and no duplicate machine IDs. Outliers are kept because failures happen at extreme readings.
Leakage fix: the five failure-type columns are removed. A depth-5 decision tree that can see them gets a fake F1 of 0.985, against an honest 0.882 without them.
New features: temperature gap (process minus air), power (torque x speed), and tool wear x torque.
Split: stratified 80% train (8,000 rows) and 20% test (2,000 rows), seed 42. The failure rate is 3.39% in train and 3.40% in test. Scaling and SMOTE run inside the pipeline, so they only learn from training data. The test set is used only for the final scores.
Models: Logistic Regression, k-NN, Decision Tree, Random Forest, LDA and AdaBoost.
Imbalance handling: three options compared per model: none, class weights and SMOTE.
Tuning: GridSearchCV with 5-fold stratified cross-validation, scored by F1 of the failure class. The best setting of each model, and the final model, are chosen on the training set only. The results below come from the notebook's quick mode (QUICK = True), which uses small search grids (for example 30 trees for Random Forest). Set QUICK = False for the full search.
Alert threshold: chosen from out-of-fold predictions on the training set, and adjustable in the app.
Results
Imbalance handling (cross-validation F1 on the training set)
Model	none	class weight	SMOTE
Random Forest	0.890	0.842	0.737
Decision Tree	0.888	0.633	0.574
AdaBoost	0.475	n/a	0.430
KNN	0.460	n/a	0.453
LDA	0.406	n/a	0.285
Logistic Regression	0.374	0.271	0.274

For every model, doing nothing about the imbalance gave the best cross-validation F1. SMOTE and class weights made the scores worse here.

Test-set scores of the best setting of each model (threshold 0.5)
Model	Imbalance	CV F1 before tuning	CV F1 after tuning	Precision	Recall	F1	PR-AUC	ROC-AUC
Random Forest	none	0.893	0.890	0.965	0.809	0.880	0.880	0.976
Decision Tree	none	0.824	0.888	0.949	0.824	0.882	0.841	0.942
AdaBoost	none	0.571	0.475	0.846	0.324	0.468	0.722	0.960
KNN	none	0.460	0.460	0.870	0.294	0.440	0.502	0.867
LDA	none	0.406	0.406	0.510	0.382	0.437	0.447	0.904
Logistic Regression	none	0.374	0.374	0.571	0.176	0.270	0.462	0.925

The tree-based models are far ahead of the linear and distance-based ones. The features most used by the Random Forest are tool wear x torque (0.226), power (0.197) and rotational speed (0.182), so two of the three engineered features are among the top three. Models are ranked by cross-validation F1, which is why Random Forest is first even though the Decision Tree's single-split test F1 is marginally higher (0.882 against 0.880).

Final model

Final model: Random Forest, no imbalance handling (30 trees, no depth limit)

Alert threshold: 0.533 (chosen on the training set)

Final test result at this threshold: precision 0.964, recall 0.794, F1 0.871, PR-AUC 0.880, ROC-AUC 0.976, accuracy 0.992.

Test set (2,000 machines)	Predicted fine	Predicted failure
Real fine	1930 (TN)	2 (FP)
Real failure	14 (FN)	54 (TP)

The model catches 54 of the 68 real failures and raises only 2 false alarms. The 14 missed failures are the main weakness. Missing a failure costs more than a false alarm, so the threshold slider in the app lets you trade some precision for higher recall.

More tables and charts are in results/ and figs/.

The app

Built with Streamlit. It loads the saved pipeline, so it uses exactly the same scaling and features as the notebook.

Page	What it does
Predict	Sliders for the readings and product type, an alert threshold slider, ready-made example machines, a red "Machine will fail!" or green "Machine is healthy" message, a probability bar and short risk notes
Model comparison	Results table and charts: before and after tuning, test scores, confusion matrices, precision-recall curves, feature importance, threshold trade-off, small decision tree
Data insights	Main EDA charts
Batch check	Upload a CSV of readings and download predictions
Run it on your computer
bash
git clone https://github.com/BhagyashreeUmap/machine_failure_predictor.git
cd machine_failure_predictor
pip install -r requirements.txt
streamlit run app.py

The app needs models/best_model.joblib, plus results/ and figs/ for its comparison and insight pages.

Repository structure
app.py                              Streamlit app
machine_failure_predictor.ipynb     full pipeline: EDA, cleaning, split, models, tuning, charts
requirements.txt                    pinned library versions
models/best_model.joblib            saved final pipeline used by the app
results/                            result tables and summary.txt
figs/                               charts used by the app and the report
docs/screenshots/                   app screenshots used in this README
README.md                           this file
Limitations
The dataset is synthetic, so scores here may be higher than on real factory data.
Each row is a single reading with no time order, so the model cannot track how a machine wears out over time.
Only 339 rows are failures, and the test set holds just 68 of them, so scores can change noticeably with a different split.
The reported tuning used small search grids (quick mode), so a wider search might find better settings.
The task is binary (failure or not); the failure type is not predicted.
The risk notes in the app are rule-of-thumb explanations from the dataset description, not part of the model.
This is an academic prototype and should not be the only basis for real maintenance decisions.
Future work
Check probability calibration, so a predicted 30% means about 30%.
Add gradient boosting (XGBoost) and SHAP explanations.
Test on real sensor data with timestamps.
Predict the failure type as a second step.
Acknowledgements

AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository (https://doi.org/10.24432/C5HS5C). Built with scikit-learn, imbalanced-learn and Streamlit.
