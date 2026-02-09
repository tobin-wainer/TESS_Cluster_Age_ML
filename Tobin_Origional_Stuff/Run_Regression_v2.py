import numpy as np
import os
from astropy.table import Table, vstack
from tensorflow import keras
from tensorflow.keras import layers
import elk  # Ensure you have the correct import for elk
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import time

start_time = time.time()

# Load and preprocess data
data = Table.read('Stats_Table_For_Data_augmented_lcs_v2.fits')
data.add_column([0 for i in range(len(data))], name='Stitched')
sector_stitched = Table.read('Stats_Table_For_Sector_Stitched.fits')
sector_stitched.add_column([1 for i in range(len(sector_stitched))], name='Stitched')
data = vstack([data, sector_stitched])

# Create labels and features
features = ['rms', 'std', 'MAD', 'sigmaG', 'von_neumann_ratio', 'J_Stetson', 'freq_at_max_power',
            'SumLSP_10_7_Day_Power', 'SumLSP_7_4_Day_Power', 'SumLSP_4_1_Day_Power', 'SumLSP_1_p5_Day_Power']

# Load and preprocess light curves
path = '/Users/Tobin/Dropbox/TESS_project/HLSPs/'
gal_list = ['MW', 'SMC', 'LMC']
l_of_cs = []

for gal in gal_list:
    filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(path + gal + "_Final_V2/")) for f in fn]
    for file in filenames:
        l_of_cs.append(elk.ensemble.from_fits(file))

l_of_all_lcs = []
for lc in l_of_cs:
    for sub_lc in lc.lcs:
        corrected_lc = sub_lc.corrected_lc
        corrected_lc['flux_err'] /= np.median(corrected_lc['flux'])
        corrected_lc['flux'] /= np.median(corrected_lc['flux'])
        l_of_all_lcs.append(corrected_lc)

# Generate sector stitched light curves
l_of_stitched_lcs = []
stitched_names = []  # To store names for sector stitched light curves
for i in range(len(l_of_cs)):
    stitched_fluxes = []
    stitched_times = []
    stitched_flux_errs = []
    for j in range(len(l_of_cs[i].lcs)):
        stitched_fluxes.append(l_of_cs[i].lcs[j].normalized_flux)
        stitched_times.append(l_of_cs[i].lcs[j].corrected_lc.time.value)
        stitched_flux_errs.append(l_of_cs[i].lcs[j].corrected_lc.flux_err.value / np.median(l_of_cs[i].lcs[j].corrected_lc.flux.value))

    catted_fluxes = np.concatenate(stitched_fluxes)
    catted_flux_errs = np.concatenate(stitched_flux_errs)
    catted_times = np.concatenate(stitched_times)
    
    stitched_table = Table([catted_times, catted_fluxes, catted_flux_errs], names=('time', 'flux', 'flux_err'))
    l_of_stitched_lcs.append(stitched_table)
    stitched_names.append(l_of_cs[i].callable)  # Assign the correct name

def find_12_day_window(light_curve):
    window_duration = 12
    total_duration = light_curve['time'][-1] - light_curve['time'][0]
    if total_duration < window_duration:
        raise ValueError("Light curve is shorter than the window duration")
    num_points = len(light_curve)
    max_start_index = num_points - int(num_points * (window_duration / total_duration.value))
    windows = [light_curve.iloc[start_index:start_index + int(num_points * (window_duration / total_duration.value))]
               for start_index in np.arange(0, (max_start_index + 1), 100)]
    return windows

data_augmented_lcs = []
for lc in l_of_all_lcs:
    data_augmented_lcs.append(lc)
    data_augmented_lcs.append(lc[:int(len(lc) / 2)])
    data_augmented_lcs.append(lc[int(len(lc) / 2):])
    data_augmented_lcs.extend(find_12_day_window(lc))

# Add sector stitched light curves to data_augmented_lcs
data_augmented_lcs.extend(l_of_stitched_lcs)

# Match light curves with data
# Assuming 'names' list matches the order of light curves with the 'data' table
names = []
for lc in l_of_cs:
    for sub_lc in lc.lcs:
        names.extend([lc.callable] * (3 + len(find_12_day_window(sub_lc.corrected_lc))))
names.extend(stitched_names)  # Add names for sector stitched light curves

# Ensure each light curve is an array of flux values
X_lc_list = [lc['flux'].data for lc in data_augmented_lcs]

# Apply median smoothing to the light curves
X_lc_smoothed_short = [median_filter(lc, size=5) for lc in X_lc_list]
X_lc_smoothed_long = [median_filter(lc, size=100) for lc in X_lc_list]

# Find the maximum length of the light curves
max_length = max(len(lc) for lc in X_lc_smoothed_short)

# Pad the light curves to the maximum length
X_lc_padded_short = np.array([np.pad(lc, (0, max_length - len(lc)), 'constant') for lc in X_lc_smoothed_short])
X_lc_padded_long = np.array([np.pad(lc, (0, max_length - len(lc)), 'constant') for lc in X_lc_smoothed_long])

# Create a mask for the padded values
mask = np.array([[1 if i < len(lc) else 0 for i in range(max_length)] for lc in X_lc_smoothed_short])

# Reshape to (samples, time_steps, 1)
X_lc_short = X_lc_padded_short.reshape((X_lc_padded_short.shape[0], X_lc_padded_short.shape[1], 1))
X_lc_long = X_lc_padded_long.reshape((X_lc_padded_long.shape[0], X_lc_padded_long.shape[1], 1))

# Ensure X_features matches the number of samples in X_lc
X_features = np.array([data[feature] for feature in features]).T[:X_lc_short.shape[0]]

# Split data based on names
unique_names = np.unique(names)
train_names, test_names = train_test_split(unique_names, test_size=0.2, random_state=42)

train_indices = np.isin(names, train_names)
test_indices = np.isin(names, test_names)

X_lc_short_train, X_lc_short_test = X_lc_short[train_indices], X_lc_short[test_indices]
X_lc_long_train, X_lc_long_test = X_lc_long[train_indices], X_lc_long[test_indices]
X_features_train, X_features_test = X_features[train_indices], X_features[test_indices]
y_train, y_test = data['Lit_Clst_Age'][train_indices], data['Lit_Clst_Age'][test_indices]
mask_train, mask_test = mask[train_indices], mask[test_indices]

# Identify and weight the 'sector_stitched' data using the 'Stitched' column
sample_weights = np.ones(len(y_train))
sample_weights[data['Stitched'][train_indices] == 1] *= 1.25  # Increase weight by 10%

# Define and compile the multi-input model for regression with masking
input_lc_short = layers.Input(shape=(X_lc_short.shape[1], 1), name='light_curve_short')
input_lc_long = layers.Input(shape=(X_lc_long.shape[1], 1), name='light_curve_long')
input_features = layers.Input(shape=(len(features),), name='features')
input_mask = layers.Input(shape=(X_lc_short.shape[1],), name='mask')

# Light curve branch with masking for short-term smoothed
x_lc_short = layers.Masking(mask_value=0.0)(input_lc_short)
x_lc_short = layers.Conv1D(32, kernel_size=3, activation='relu')(x_lc_short)
x_lc_short = layers.MaxPooling1D(pool_size=2)(x_lc_short)
x_lc_short = layers.Conv1D(64, kernel_size=3, activation='relu')(x_lc_short)
x_lc_short = layers.MaxPooling1D(pool_size=2)(x_lc_short)
x_lc_short = layers.Flatten()(x_lc_short)

# Light curve branch with masking for long-term smoothed
x_lc_long = layers.Masking(mask_value=0.0)(input_lc_long)
x_lc_long = layers.Conv1D(32, kernel_size=3, activation='relu')(x_lc_long)
x_lc_long = layers.MaxPooling1D(pool_size=2)(x_lc_long)
x_lc_long = layers.Conv1D(64, kernel_size=3, activation='relu')(x_lc_long)
x_lc_long = layers.MaxPooling1D(pool_size=2)(x_lc_long)
x_lc_long = layers.Flatten()(x_lc_long)

# Combine branches
x = layers.concatenate([x_lc_short, x_lc_long, input_features])
x = layers.Dense(100, activation='relu')(x)
output = layers.Dense(1)(x)  # Regression output

model = keras.Model(inputs=[input_lc_short, input_lc_long, input_features, input_mask], outputs=output)

# Custom loss function to incorporate priors
def custom_loss(y_true, y_pred):
    prior_mean = 8.35
    prior_std = 1.85
    prior = keras.losses.MeanSquaredError()(y_pred, prior_mean)
    return keras.losses.MeanSquaredError()(y_true, y_pred) + prior_std * prior

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=custom_loss,
    metrics=['mae']
)

# Train the model
history = model.fit(
    {'light_curve_short': X_lc_short_train, 'light_curve_long': X_lc_long_train, 'features': X_features_train, 'mask': mask_train},
    y_train,
    sample_weight=sample_weights,
    batch_size=32,
    epochs=100,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, verbose=1, min_delta=1e-5),
        keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=10, verbose=1)
    ],
    verbose=1
)

# Evaluate the model
test_loss, test_mae = model.evaluate(
    {'light_curve_short': X_lc_short_test, 'light_curve_long': X_lc_long_test, 'features': X_features_test, 'mask': mask_test},
    y_test
)
print(f'Test MAE: {test_mae}')


# Plot predicted vs actual values for the test set
y_pred = model.predict({'light_curve_short': X_lc_short_test, 'light_curve_long': X_lc_long_test, 'features': X_features_test, 'mask': mask_test})
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.savefig('out_figs/Predicted_vs_Lit.png')
plt.show()
            

# Estimate uncertainty using Monte Carlo Dropout
def mc_dropout_predict(model, inputs, n_iter=100):
    f = keras.backend.function([model.input], [model.output])
    result = [f(inputs) for _ in range(n_iter)]
    return np.array(result).squeeze()

y_pred_mc = mc_dropout_predict(model, {'light_curve_short': X_lc_short_test, 'light_curve_long': X_lc_long_test, 'features': X_features_test, 'mask': mask_test})
y_pred_mean = y_pred_mc.mean(axis=0)
y_pred_std = y_pred_mc.std(axis=0)

# Plot the uncertainty
plt.figure(figsize=(12, 6))
plt.errorbar(y_test, y_pred_mean, yerr=y_pred_std, fmt='o', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title('Predicted vs Actual Values with Uncertainty')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.savefig('out_figs/Uncertainty.png')
plt.show()


t = Table([y_test, y_pred, y_pred_mean, y_pred_std], names=('Test_age', 'Predicted_Age', 'Pred_mc_Mean', 'Pred_mc_STD'))
t.write('out_figs/Results.fits')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken to run: {elapsed_time:.3f} seconds")
