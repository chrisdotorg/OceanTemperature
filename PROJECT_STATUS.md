## Regression Problem

### Target Variable

`T_degC` : water temperature in degrees Celsius.

### Objective

Build a linear regression model to predict ocean water temperature
from selected oceanographic, geographic, and temporal features in
the CalCOFI Bottle dataset.

### Primary Evaluation Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² score

### Quality Control

CalCOFI quality codes will be considered during data preparation:

- `6`: valid data originating from a CTD sensor
- `8`: questionable data
- `9`: missing data

Observations with missing target values will not be used for supervised
training. Questionable temperature measurements will be investigated
before the final training dataset is constructed.

## Model Evaluation

### Test Set

The final model was evaluated on a held-out test set containing
88,683 observations.

### Baseline

A mean-temperature baseline was established by predicting the
mean temperature observed in the training set for every test
observation.

The baseline achieved:

- MAE: 3.0894 °C
- MSE: 13.8199
- RMSE: 3.7175 °C
- R²: -0.0083

### PyTorch Linear Regression

The final PyTorch linear regression model was trained using
eight standardized features:

- `Depthm`
- `Salnty`
- `O2ml_L`
- `Year`
- `Lat_Dec`
- `Lon_Dec`
- `Month_sin`
- `Month_cos`

The model achieved on the held-out test set:

- MAE: 1.3253 °C
- MSE: 3.1896
- RMSE: 1.7859 °C
- R²: 0.7673

### Improvement Over Baseline

Compared with the mean-temperature baseline:

- MAE improved by 57.10%
- RMSE improved by 51.96%

The results demonstrate that the selected features provide
substantial predictive information about ocean water temperature.

### Error Analysis

The error analysis revealed that prediction performance is not
uniform across the dataset.

The model performs relatively well for the most represented
temperature ranges, particularly between 5 °C and 15 °C.

Larger errors were observed for extreme temperature values,
especially observations above 20 °C and below 5 °C.

Depth-based analysis also revealed larger errors for observations
deeper than 1000 meters. This range contains substantially fewer
observations than the dominant 100–500 meter range.

The largest prediction errors were associated with extreme
temperature observations, indicating that the linear model has
difficulty representing some nonlinear relationships present in
the oceanographic data.

### Interpretation

The model provides a strong improvement over the mean baseline
and explains approximately 76.7% of the variance in the test-set
temperature observations.

However, the results should not be interpreted as evidence that
the relationship between the selected features and temperature is
strictly linear. The residual analysis suggests nonlinear
structure and systematic prediction errors, particularly at
extreme temperatures and depths.

These limitations will be discussed in the final report.