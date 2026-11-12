'''
Developed using code from Farisa Morales et al's SED Fitter
and assistance from GitHub Copilot

Sydney Sk 2026
'''

import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

VERBOSE_BOOL = False
SHOW_PLOTS_BOOL = False

BAND_WAVELENGTHS = {
    "2MASS J": 1.235,
    "2MASS H": 1.662,
    "2MASS K": 2.159,
    "Spitzer I1": 3.6,
    "Spitzer I2": 4.5,
    "Spitzer I3": 5.8,
    "Spitzer I4": 8.0,
    "Spitzer M24": 24.0,
    "Spitzer M70": 70.0,
    "Spitzer M160": 160.0,
    "WISE W1": 3.368,
    "WISE W2": 4.618,
    "WISE W3": 12.082,
    "WISE W4": 22.194,
}

ZERO_POINT_FLUXES = {
    "2MASS J": 1594.0,
    "2MASS H": 1024.0,
    "2MASS K": 666.7,
    "Spitzer I1": 280.9,
    "Spitzer I2": 179.7,
    "Spitzer I3": 115.0,
    "Spitzer I4": 64.13,
    "Spitzer M24": 7.17,
    "Spitzer M70": 0.778,
    "Spitzer M160": 0.159,
    "WISE W1": 309.54, # Uncertain if appropriate values for WISE
    "WISE W2": 171.787,
    "WISE W3": 31.676,
    "WISE W4": 8.3635,
}

# band and columns from files containing wavelengths and filter response functions (FRFs)
FRF_FILES = {
        "2MASS J": ("2massj.txt", (1, 2)),
        "2MASS H": ("2massh.txt", (1, 2)),
        "2MASS K": ("2massk.txt", (1, 2)),
        "Spitzer I1": ("FullArrayAverageSpecResp_I1.txt", (0, 1)),
        "Spitzer I2": ("FullArrayAverageSpecResp_I2.txt", (0, 1)),
        "Spitzer I3": ("FullArrayAverageSpecResp_I3.txt", (0, 1)),
        "Spitzer I4": ("FullArrayAverageSpecResp_I4.txt", (0, 1)),
        "Spitzer M24": ("MIPSfiltsumm24.txt", (0, 1)),
        "Spitzer M70": ("MIPSfiltsumm70.txt", (0, 1)),
        "Spitzer M160": ("MIPSfiltsumm160.txt", (0, 1)),
        "WISE W1": ("w1rsr.tbl", (0, 1)),
        "WISE W2": ("w2rsr.tbl", (0, 1)),
        "WISE W3": ("w3rsr.tbl", (0, 1)),
        "WISE W4": ("w4rsr.tbl", (0, 1)),
    }

def load_filter_response_functions(FRF_folder):
    """Load, remove duplicates, and normalize all filter-response functions.

    Parameters
    ----------
    FRF_folder : pathlib.Path
        Folder containing the filter-response files.

    Returns
    -------
    dict
        Maps each band name to a ``(wavelengths, filter response function)`` tuple.
    """

    # Initialize the FRF dictionary
    FRFs = {}

    # Loop through each filter
    for name, (filename, usecols) in FRF_FILES.items():
        
        # Load text according to columns specified in FRF_FILES, unpacking into two arrays: wavelengths and responses
        waves, responses = np.loadtxt(
            FRF_folder / filename,
            unpack=True,
            usecols=usecols,
        )

        # Remove duplicates - keep the first response for each unique wavelength
        unique_waves, first_indices = np.unique(
            waves,
            return_index=True,
        )
        responses = responses[first_indices]

        # Normalize the filter response function to a maximum of 1
        FRFs[name] = (unique_waves, responses / responses.max())

    return FRFs

def measure_flux(star_waves, star_fluxes, FRFs):
    """Measure response-weighted fluxes for every filter.

    Parameters
    ----------
    star_waves : numpy.ndarray
        Stellar-model wavelengths in microns.
    star_fluxes : numpy.ndarray
        Stellar-model flux densities per star_wave.
    FRFs : dict
        Mapping of band names to ``(filter_waves, response)`` arrays.

    Returns
    -------
    dict
        Maps each band name to its response-weighted flux.
    """
    return {
        band: simpson(
            np.interp(filter_waves, star_waves, star_fluxes) * response,
            x=filter_waves,
        ) / simpson(response, x=filter_waves)
        for band, (filter_waves, response) in FRFs.items()
    }

def measure_magnitudes(band_fluxes):
    """Convert measured band fluxes into magnitudes.

    Parameters
    ----------
    band_fluxes : dict
        Mapping of band names to response-weighted fluxes in Jy.

    Returns
    -------
    dict
        Mapping of band names to magnitudes. Invalid fluxes produce ``NaN``.
    """
    return {
        band: (
            -2.5 * np.log10(flux / ZERO_POINT_FLUXES[band])
            if np.isfinite(flux) and flux > 0
            else np.nan
        )
        for band, flux in band_fluxes.items()
    }
    
def stellar_model_measurements(model_file, FRFs):
    """Measure band fluxes and magnitudes for one stellar-model file.

    Parameters
    ----------
    model_file : pathlib.Path
        Stellar-model file containing wavelength and ``F_lambda`` columns.
    FRFs : dict
        Mapping of band names to ``(filter_waves, response)`` arrays.

    Returns
    -------
    list or None
        Magnitudes in the order of ``FRFs``, or ``None`` if processing fails.
    """
    
    if VERBOSE_BOOL:
        print(f"Processing model file: {model_file.name}")
    
    # Concerning the nextgen and kurucz models: the wavelengths are in units of
    # angstrom, which equals 10^-10 meters. Flux is in erg/cm^2/s/a - at stellar surface?
    duplicate_waves, duplicate_fluxes = np.loadtxt(model_file,unpack = True)

    # Remove duplicates if present
    waves_Angstrom, indices = np.unique(duplicate_waves, return_index = True)
    fluxes_F_lambda = duplicate_fluxes[indices]

    # Check if fluxes within stellar model are finite and positive - remove any invalid fluxes and corresponding wavelengths
    invalid_samples = ~np.isfinite(fluxes_F_lambda) | (fluxes_F_lambda <= 0)
    invalid_waves_um = waves_Angstrom[invalid_samples] * 1e-4

    # Identify bands whose filter response functions contain invalid wavelengths
    invalid_bands = []

    # Loop through each band/FRF
    for band, (filter_waves, _) in FRFs.items():
        # True if any invalid wavelengths fall within the filter's wavelength range
        in_filter = (
            (invalid_waves_um >= filter_waves.min())
            & (invalid_waves_um <= filter_waves.max())
        )

        # If True, add band to list
        if in_filter.any():
            invalid_bands.append(band)

    if VERBOSE_BOOL:
        for band in invalid_bands:
            print(
                f"Warning: dropping {band} - contains 0.0 spectral fluxes in stellar model")

    waves_Angstrom = waves_Angstrom[~invalid_samples]
    fluxes_F_lambda = fluxes_F_lambda[~invalid_samples]

    # Convert stellar model F_lambda to F_nu and to Jy (Janskys)
    c_A_Hz = 2.99792458e18 # speed of light in Angstrom * Hz
    waves_um = waves_Angstrom * 1e-4 # Angstroms (10^-10 m) to um (10^-6 m) = 1.0e-4, NOT cm -> um (microns)
    fluxes_Jy = (fluxes_F_lambda * waves_Angstrom ** 2 /c_A_Hz) * 1.0e23

    try:
        # Measure band fluxes first, then convert those results to magnitudes.
        band_fluxes = measure_flux(waves_um, fluxes_Jy, FRFs)
        for band in invalid_bands:
            band_fluxes[band] = np.nan
        magnitudes = measure_magnitudes(band_fluxes)

        if SHOW_PLOTS_BOOL:
            figure, axis = plt.subplots(figsize=(12, 7))
            axis.plot(waves_um, fluxes_Jy, label="Stellar model")

            for band, band_flux in band_fluxes.items():
                band_wavelength = BAND_WAVELENGTHS[band]
                axis.scatter(band_wavelength, band_flux, s=35)
                axis.annotate(
                    band,
                    (band_wavelength, band_flux),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

            axis.set_yscale("log")
            axis.set_xlabel("Wavelength (microns)")
            axis.set_ylabel("Flux density (Jy)")
            axis.set_title(f"{model_file.name}: model spectrum and band fluxes")
            axis.grid(True, which="both", alpha=0.25)
            figure.tight_layout()
            plt.show()
            plt.close('all')

        return list(magnitudes.values())

    except:
        return None

# Execute if the script is run directly (not imported as a module).
if __name__ == "__main__":

    # Define the path to the folder containing stellar model files.
    cwd = Path(__file__).parent
    stellar_models_folder = cwd / "StellarModels"
    FRFs_folder = cwd / "FilterResponseFunctions"

    # Load the Filter Response Functions (FRFs) for each filter.
    FRF_data = load_filter_response_functions(FRFs_folder)

    '''
    # Measure the magnitudes for a specific stellar model.
    placeholder_dummy_var = stellar_model_measurements(
        stellar_models_folder / "Kurucz/km01_10000g40.txt",
        FRF_data)
    '''

    # Load all stellar model files recursively from the specified folder's subdirectories.
    model_files = [f for f in stellar_models_folder.rglob("*.txt") if f.is_file()]

    # Read or create an output file to record unnormalized magnitudes for each stellar model and failed models (all 0.0).
    output_file = cwd / "unnormalized_magnitudes.csv"
    failed_file = cwd / "failed_models.txt"

    # Define the column names for the output DataFrame.
    magnitude_columns = ["J", "H", "K", "I1", "I2", "I3", "I4",
                         "M24", "M70", "M160", "W1", "W2", "W3", "W4"]

    # Read the existing output file if it exists, otherwise create a new DataFrame with the specified columns.
    if output_file.exists():
        df = pd.read_csv(output_file)
    else:
        df = pd.DataFrame(columns=["Model Name", *magnitude_columns])

    # Read the existing failed models file if it exists, otherwise create an empty set to track failed models.
    failed_models = set()
    if failed_file.exists():
        failed_models = set(failed_file.read_text(encoding="utf-8").splitlines())

    # Loop through each stellar model file, measure its magnitudes per band, and record the results in the output file.
    for model_file in model_files:

        # File name without path
        model_name = model_file.name

        # Skip if previously ran (found in file)
        if model_name in df["Model Name"].values:
            if VERBOSE_BOOL:
                print(f"Skipping existing model: {model_name}. Already processed.")
            continue

        # Measure unnormalized magnitudes for the current stellar model.
        magnitudes = stellar_model_measurements(model_file, FRF_data)

        # If error, record to failed models file and continue
        if magnitudes is None:
            if VERBOSE_BOOL:
                print(f"Failed to measure magnitudes for model: {model_name}.")
            if model_name not in failed_models:
                with failed_file.open("a", encoding="utf-8") as file:
                    file.write(f"{model_name}\n")
                failed_models.add(model_name)
            continue

        # Append the new model to output csv
        df.loc[len(df)] = [model_name, *magnitudes]
        df.to_csv(output_file, index=False)