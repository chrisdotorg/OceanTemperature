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