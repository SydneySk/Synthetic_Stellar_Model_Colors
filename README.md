# Synthetic Model Color Calculations for 2MASS, Spitzer, and WISE

This program calculates stellar-model fluxes and magnitudes *I BELIEVE AT THE SURFACE OF THE STAR* for
2MASS, Spitzer, and WISE bands, then records the results in a CSV file.

Intended for color calculations, hence disregard for distance/flux normalization.

## Inputs

- Stellar models are read from `StellarModels/` 
- Model wavelengths are expected in Angstroms.
- Filter response files are read from `FilterResponseFunctions/`.
- Filter-response wavelengths are expected in microns.

Supported bands are:

- 2MASS: J, H, K
- Spitzer: I1, I2, I3, I4, M24, M70, M160
- WISE: W1, W2, W3, W4

## Processing

1. Duplicate wavelengths in each filter-response file are reduced by keeping
	the first response value.
2. Each response curve is normalized to a maximum of 1. This normalization
	does not change the response-weighted mean because the result is divided by
	the integrated response.
3. Stellar-model wavelengths are converted from Angstroms to microns.
4. Stellar `F_lambda` values are converted to `F_nu` in Jy.
5. The model is interpolated onto each filter-response wavelength grid.
6. Each band flux is calculated as:

	`integral(model_flux * response) / integral(response)`

7. Magnitudes are calculated using the published zero-point flux for each
	band:

	`magnitude = -2.5 * log10(band_flux / zero_point_flux)`

The results are synthetic, response-weighted colors. They are intended for
comparison of model color trends.

## Invalid Fluxes

Nonfinite and nonpositive model-flux samples are removed. If any such sample
falls within a filter's wavelength range, that entire band is recorded as
`NaN` rather than being interpolated across.

## Outputs

- `unnormalized_magnitudes.csv` stores one row per successfully processed
  stellar model.
- `failed_models.txt` stores model names that could not be processed.
- Set `SHOW_PLOTS_BOOL = True` to display the stellar spectrum and the
  response-weighted band fluxes at the defined band wavelengths.
- Set `VERBOSE_BOOL = True` to print status messages