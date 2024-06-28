import numpy as np
from scipy.interpolate import interp1d

def compute_dvh(mask, dose, minDose = 0.0, maxDose = None, binSize = 0.2, max_allowed_dose = 100.0):
    """Computes the DVH corresponding to the `dose` inside of a `mask`. The 
       output are the dose and fractional volume pairs for a 2D DVH plot.
       
    Created by Ivan Vazquez in collaboration with Ming Yang.
       
    Parameters:
    -----------
        mask (ndarray): 3D mask volume of the VOI where the DVH will be computed
        dose (ndarray): Volumetric dose distribution used to compute the DVH
        minDose (float, optional): minimum value for the dose axis. Defaults to 0.
        maxDose (float, optional): maximum value for the dose axis. Defaults to None.
        binSize (float, optional): Size of the dose binss. Defaults to 0.2.
        max_allowed_dose (float, optional): Maximum dose value to consider. Defaults to 100.0.

    Returns:
    --------
        tupe: pair of arrays containing the dose and volume values.
    """
    
    # COMPUTE DVH 
    ## 1. Mask dose volume with 3d contour 
    maskedDose = np.multiply(mask,dose)
    
    # 1.1 Check if the volume is empty (all zeros). If so, return 0's for contour
    if np.round(maskedDose,5).max() == 0.0: return np.array([0]),np.array([0])
        
    ## 2. Find indices of voxels inside of masked VOI (values in mask greater than 0)
    inds = np.where(mask > 0)

    ## 3. Extract values of dose inside of VOI 
    doseInVOI = maskedDose[inds]

    ## 4. Define maximum dose value and get the number of voxels in VOI volume
    maxDoseVal, numVox = maxDose, np.sum(doseInVOI.shape)
    if maxDoseVal is None: maxDoseVal = np.ceil(doseInVOI.max())
    
    # Make sure that the maximum dose value is not greater than the maximum allowed dose
    if maxDoseVal > max_allowed_dose: maxDoseVal = max_allowed_dose

    ## 5. Dose values to inspect - create abscissa 
    d = np.linspace(minDose, maxDoseVal, int(np.ceil(maxDoseVal+1)/binSize), endpoint=True)

    ## 6. Find percentage of volume with dose above
    v = np.zeros(len(d))
    for n, dv in enumerate(d):
        v[n] = 100*len(np.where(doseInVOI >= dv)[0])/numVox
        
    return d, v

def D(volume, d,v):
    try:
        D = interp1d(np.flip(v[np.argmin(v==100.0)-1:]), np.flip(d[np.argmin(v==100.0)-1:]), bounds_error=False, fill_value="extrapolate")
        return float(D(volume))
    except:
        return 0

def V(dose, d,v):
    try:
        V = interp1d(d, v, bounds_error=False, fill_value=0)
        return float(V(dose))
    except:
        return 0