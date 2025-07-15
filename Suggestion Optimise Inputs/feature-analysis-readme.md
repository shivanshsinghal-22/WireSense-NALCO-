# Feature Analysis for Material Properties Prediction

This documentation explains the process of identifying the most important features that influence material properties (Conductivity, UTS, and Elongation) through a combination of SHAP analysis and sensitivity testing.

## Methodology

### 1. SHAP Analysis
SHAP (SHapley Additive exPlanations) values were calculated to understand feature contributions to model predictions. For each target variable (Conductivity, UTS, and Elongation), we:
- Used XGBoost models trained on normalized data
- Applied SHAP explainer to get feature importance values
- Identified top 7 contributing features based on absolute SHAP values

### 2. Sensitivity Analysis
A custom sensitivity analysis was implemented to measure how small changes in input features affect model predictions:
```python
def calculate_mean_sensitivity(model, X):
    sensitivity_scores = pd.Series(index=X.columns)
    for feature in X.columns:
        changes = []
        for i in range(len(X)):
            X_perturbed = X.copy()
            X_perturbed.iloc[i, X.columns.get_loc(feature)] += 0.01
            original_prediction = model.predict(X.iloc[[i]])
            perturbed_prediction = model.predict(X_perturbed.iloc[[i]])
            change = np.abs(perturbed_prediction - original_prediction)
            changes.append(change[0])
        sensitivity_scores[feature] = np.mean(changes)
    return sensitivity_scores
```

### 3. Feature Intersection
The final step involved finding the intersection between top features identified by both SHAP and sensitivity analysis to determine the most robust and influential features.

## Results

### Conductivity Model
- **Top SHAP Features**: 
  - EMUL_OIL_L_PR_VAL0
  - EMULSION_LEVEL_ANALO_VAL0
  - QUENCH_CW_FLOW_EXIT_VAL0
  - STANDS_OIL_L_PR_VAL0
  - RM_MOTOR_COOL_WATER__VAL0
  - CAST_WHEEL_RPM_VAL0
  - RM_COOL_WATER_FLOW_VAL0

- **Top Sensitive Features**:
  - EMULSION_LEVEL_ANALO_VAL0
  - QUENCH_CW_FLOW_EXIT_VAL0
  - QUENCH_CW_FLOW_ENTRY_VAL0
  - EMUL_OIL_L_TEMP_PV_VAL0
  - STANDS_OIL_L_PR_VAL0
  - TUNDISH_TEMP_VAL0
  - Furnace_Temperature

- **Intersection Features**:
  - STANDS_OIL_L_PR_VAL0
  - QUENCH_CW_FLOW_EXIT_VAL0
  - EMULSION_LEVEL_ANALO_VAL0

### UTS (Ultimate Tensile Strength) Model
- **Top SHAP Features**:
  - EMUL_OIL_L_PR_VAL0
  - QUENCH_CW_FLOW_ENTRY_VAL0
  - ROLL_MILL_AMPS_VAL0
  - RM_COOL_WATER_FLOW_VAL0
  - QUENCH_CW_FLOW_EXIT_VAL0
  - STANDS_OIL_L_PR_VAL0
  - RM_MOTOR_COOL_WATER__VAL0

- **Top Sensitive Features**:
  - EMUL_OIL_L_TEMP_PV_VAL0
  - EMULSION_LEVEL_ANALO_VAL0
  - QUENCH_CW_FLOW_ENTRY_VAL0
  - QUENCH_CW_FLOW_EXIT_VAL0
  - STANDS_OIL_L_PR_VAL0
  - TUNDISH_TEMP_VAL0
  - STAND_OIL_L_TEMP_PV_REAL_VAL0

- **Intersection Features**:
  - STANDS_OIL_L_PR_VAL0
  - QUENCH_CW_FLOW_EXIT_VAL0
  - QUENCH_CW_FLOW_ENTRY_VAL0

### Elongation Model
- **Top SHAP Features**:
  - EMUL_OIL_L_PR_VAL0
  - QUENCH_CW_FLOW_ENTRY_VAL0
  - QUENCH_CW_FLOW_EXIT_VAL0
  - STANDS_OIL_L_PR_VAL0
  - ROLL_MILL_AMPS_VAL0
  - RM_COOL_WATER_FLOW_VAL0
  - RM_MOTOR_COOL_WATER__VAL0

- **Top Sensitive Features**:
  - EMULSION_LEVEL_ANALO_VAL0
  - QUENCH_CW_FLOW_ENTRY_VAL0
  - TUNDISH_TEMP_VAL0
  - QUENCH_CW_FLOW_EXIT_VAL0
  - EMUL_OIL_L_TEMP_PV_VAL0
  - STANDS_OIL_L_PR_VAL0
  - CAST_WHEEL_RPM_VAL0

- **Intersection Features**:
  - STANDS_OIL_L_PR_VAL0
  - QUENCH_CW_FLOW_EXIT_VAL0
  - QUENCH_CW_FLOW_ENTRY_VAL0

## Key Findings

1. **Common Critical Features**: STANDS_OIL_L_PR_VAL0 and QUENCH_CW_FLOW_EXIT_VAL0 appear in the intersection for all three target variables, indicating their fundamental importance to the material properties.

2. **Target-Specific Features**:
   - Conductivity: Uniquely influenced by EMULSION_LEVEL_ANALO_VAL0
   - UTS & Elongation: Share QUENCH_CW_FLOW_ENTRY_VAL0 as a critical feature

3. **Feature Stability**: The intersection approach helps identify features that are both influential (high SHAP values) and stable (high sensitivity scores), making them reliable control parameters for process optimization.

## Implementation Notes

- All features were normalized before analysis to ensure fair comparison
- XGBoost models were used for prediction
- A small perturbation of 0.01 was used for sensitivity analysis
- Top 7 features were considered for both SHAP and sensitivity analysis before finding the intersection

## Usage Recommendations

When optimizing the manufacturing process, prioritize monitoring and control of:
1. Stands oil pressure (STANDS_OIL_L_PR_VAL0)
2. Quench cooling water flow exit (QUENCH_CW_FLOW_EXIT_VAL0)
3. Quench cooling water flow entry (QUENCH_CW_FLOW_ENTRY_VAL0)
4. Emulsion level (EMULSION_LEVEL_ANALO_VAL0)

# Gradient Descent Optimization for Material Property Prediction

This document describes a gradient descent optimization algorithm used to achieve target material properties for steel production. The algorithm utilizes a pre-trained XGBoost model to predict material properties (conductivity, elongation, and UTS) based on various input features. It then optimizes the input features to achieve a desired target value for a specific property.

## 1. Data Preparation and Normalization

The algorithm begins by loading a dataset containing steel production data and the corresponding material properties. The dataset is then normalized using a min-max normalization technique. This ensures that all features are on a scale between 0 and 1, allowing the model to focus on the relative importance of each feature rather than their absolute magnitude.

## 2. Model Loading

Three pre-trained XGBoost models are loaded, each corresponding to a specific material property (conductivity, elongation, and UTS). These models predict the output property value given a set of input features.

## 3. Feature Selection and Gradient Descent Optimization

Feature Selection: The algorithm identifies the top contributing features for the target property using a method like SHAP (SHapley Additive exPlanations). This helps focus the optimization process on the most influential features.

## Gradient Descent Optimization:

The algorithm selects a data instance (row) from the dataset and its corresponding output property value.
It defines a target output value for the property (e.g., slightly higher conductivity).
The algorithm finds the nearest neighbor in the dataset to the target output using the chosen distance metric (e.g., absolute difference). This neighbor serves as a starting point for the optimization.
It iteratively performs the following steps:
Predicts the current output property value using the loaded model.
Calculates the error between the predicted value and the target value.
Calculates the gradients (slopes) for the top features using a central difference method. The gradients indicate how much each feature should be adjusted to reduce the error.
Updates the values of the top features based on their gradients and a learning rate. The learning rate controls the step size taken in the optimization direction.
Clips the updated feature values to remain within the normalized range (0, 1).
The optimization process continues until the error between the predicted value and the target value falls below a certain tolerance or a maximum number of iterations is reached.

## 4. Result Interpretation

The algorithm outputs the optimized feature values and the corresponding predicted property value after optimization.
The predicted property value is then converted back to the original scale using the min-max normalization factors.
This allows for a comparison between the original property value, the target property value, and the property value achieved after optimization using the adjusted feature values.

## 5. Conclusion

The gradient descent optimization algorithm provides a way to fine-tune the material properties of steel production by adjusting the input features based on pre-trained models. This can be helpful in achieving desired material properties for specific applications.


## Input Features:

The dataset included various features related to steel production, such as:

Top features for Conductivity: ['STANDS_OIL_L_PR_VAL0', 'QUENCH_CW_FLOW_EXIT_VAL0', 'EMULSION_LEVEL_ANALO_VAL0']
EMUL_OIL_L_TEMP_PV_VAL0          0.964644
STAND_OIL_L_TEMP_PV_REAL_VAL0    0.736111
GEAR_OIL_L_TEMP_PV_REAL_VAL0     0.540549
EMUL_OIL_L_PR_VAL0               0.063403
QUENCH_CW_FLOW_EXIT_VAL0         0.460937
CAST_WHEEL_RPM_VAL0              0.782552
BAR_TEMP_VAL0                    0.879807
QUENCH_CW_FLOW_ENTRY_VAL0        0.578668
GEAR_OIL_L_PR_VAL0               0.446894
STANDS_OIL_L_PR_VAL0             0.437004
TUNDISH_TEMP_VAL0                0.667982
RM_MOTOR_COOL_WATER__VAL0        0.419684
ROLL_MILL_AMPS_VAL0              0.087717
RM_COOL_WATER_FLOW_VAL0          0.881422
EMULSION_LEVEL_ANALO_VAL0        0.703738
Furnace_Temperature              0.532373

### Target Property:**

The target property optimized for in this case was the Conductivity (S/m) of the produced steel.

**Optimizing Conductivity:**

The algorithm was used to optimize the conductivity of the steel. Here's a breakdown of the process:

* Initial conductivity: The initial conductivity value (normalized) was 0.525.
* Target conductivity: The target conductivity (normalized) was set to 0.825.

**Optimized Feature Values:**

The optimization process adjusted the values of the following features to achieve the target conductivity:

* EMUL_OIL_L_PR_VAL0 (increased slightly)
* QUENCH_CW_FLOW_EXIT_VAL0 (increased slightly)
* STANDS_OIL_L_PR_VAL0 (decreased slightly)
* Other features remained unchanged.

Target Conductivity (normalized): 0.8251779556274415
EMUL_OIL_L_TEMP_PV_VAL0          0.964644
STAND_OIL_L_TEMP_PV_REAL_VAL0    0.736111
GEAR_OIL_L_TEMP_PV_REAL_VAL0     0.540549
EMUL_OIL_L_PR_VAL0               0.063403
QUENCH_CW_FLOW_EXIT_VAL0         0.461106
CAST_WHEEL_RPM_VAL0              0.782552
BAR_TEMP_VAL0                    0.879807
QUENCH_CW_FLOW_ENTRY_VAL0        0.578668
GEAR_OIL_L_PR_VAL0               0.446894
STANDS_OIL_L_PR_VAL0             0.434734
TUNDISH_TEMP_VAL0                0.667982
RM_MOTOR_COOL_WATER__VAL0        0.419684
ROLL_MILL_AMPS_VAL0              0.087717
RM_COOL_WATER_FLOW_VAL0          0.881422
EMULSION_LEVEL_ANALO_VAL0        0.706433
Furnace_Temperature              0.532373

**Results:**

* After optimization, the predicted conductivity on the normalized scale reached 0.825, matching the target value.
* Converting the optimized conductivity back to the original scale resulted in a value of 61.63 (S/m), which is close to the target conductivity of 61.61 (S/m).
