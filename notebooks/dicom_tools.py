from pydicom import dcmread
from scipy.interpolate import interp1d
from skimage.draw import polygon
from dataclasses import dataclass, field
from utilities import interpolate_volume, natural_keys
from tqdm import tqdm
import multiprocessing as mp

import os, json, copy, pandas, h5py, logging, re, pickle, traceback, statistics

import numpy as np


# Known issues:
# 1. Radiation type is expected. Make this optional.

@dataclass
class Coordinates():
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    dx: float
    dy: float
    dz: float
    image_position: np.ndarray

@dataclass
class CT():
    shape: tuple  
    resolution: np.ndarray
    max_value: float
    min_value: float  
    units: str 
    rescale_slope: float
    rescale_intercept: float
    patient_position: str
    data: np.ndarray
    slice_thickness: float
    coordinates: Coordinates
    
@dataclass
class CommonDoseTags():
    shape: tuple
    max_value: float
    min_value: float
    resolution: np.ndarray
    dose_grid_scaling: float
    dose_units: str
    data: np.ndarray
    coordinates: Coordinates
    beam_number: int = 1
    beam_type: str = 'not_specified'
    gantry_angle: float = 0.0
    patient_support_angle: float = 0.0
    table_top_pitch_angle: float = 0.0
    table_top_roll_angle: float = 0.0
    isocenter: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    radiation_type: str = 'not_specified'
    treatment_delivery_type: str = 'not_specified'
    beam_name: str = 'not_specified'
    treatment_machine: str = 'not_specified'
    beam_description: str = 'not_specified'
    number_of_control_points: int = 0
    final_cumulative_meterset_weight: float = 0.0
    beam_dose: float = 0.0
    scan_mode: str = 'not_specified'
    primary_dosimetric_units: str = 'not_specified'
     
@dataclass
class ProtonDose(CommonDoseTags):
    vsad: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    
@dataclass
class PhotonDose(CommonDoseTags):
    sad: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    gantry_rotation_direction: list = field(default_factory=lambda: ['not_specified'])
    
@dataclass
class Mask():
    data: np.ndarray
    resolution: str
    coordinates:Coordinates
    number: int
    color: str = 'not_specified'
    roi_genration_algorithm: str = 'not_specified'
    
@dataclass
class Plan():
    number_of_beams: int = 1
    geometry: str = 'not_specified'
    patient_position: str = 'not_specified'
    patient_sex: str = 'not_specified'
    plan_label: str = 'not_specified'
    number_of_fractions_planned: int = 0
    dose_per_fraction: float = 0.0
    dose_reference_type: str = 'not_specified'
    dose_reference_description: str = 'not_specified'
    dose_reference_dose: float = 0.0
    radiation_type: str = 'not_specified'
    beam: dict = field(default_factory=lambda: {})
    
@dataclass
class Beam():
    gantry_angle: float = 0.0
    patient_support_angle: float = 0.0
    isocenter: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    treatment_delivery_type: str = 'not_specified'
    treatment_machine: str = 'not_specified'
    type: str = 'not_specified'
    sad: float = 0.0
    vsad: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))

class DicomToolbox():
    """"Class for parsing a set of DICOM-RT files for a patient.

        Created on the Fall of 2021 by Ivan Vazquez in collaboration with Ming Yang. 

        Last updated: January 2024

        Copyright 2021-2024 Ivan Vazquez
    """
    
    def __init__(self, user_inputs_dir=None, patient_data_directory=None, lut_directory=None) -> None:
        
        self.__logger = logging.getLogger(__name__)

        # Prepare RSP LUT directory
        if lut_directory is not None:
            lut_directory = os.path.join('utilities','LUT')
            if not os.path.isdir(lut_directory): 
                self.__logger.warning(f'The directory {lut_directory} for the look-up tables does not exist.')
                self.lut_directory = None
            else:
                self.lut_directory = lut_directory

        # Load user input information if a directory is provided
        if user_inputs_dir is not None:
            with open(user_inputs_dir, "r") as f:
                self.user_inputs = json.load(f)  
                self.patient_data_directory = self.user_inputs['DIRECTORIES']['raw_patient_data']
        else:
            self.__set_default_user_inputs()
            
        # set the directory for the patient data to the specified value if one is given
        if patient_data_directory is not None:
            assert os.path.isdir(patient_data_directory), f"The directory {patient_data_directory} does not exist."
            self.patient_data_directory = patient_data_directory 
            self.user_inputs['DIRECTORIES']['raw_patient_data'] = patient_data_directory 

        # Check if the necessary directories exists
        for directory in ['logs', 'temp', os.path.join('temp','data')]:
            if not os.path.isdir(directory): os.makedirs(directory, exist_ok=True)
            
        # Initialize variables
        self.reset()
        self.parallelize = None
        self.n_threads = self.user_inputs["PARALLELIZATION"]["number_of_processors"]
        self.min_coordinate_precision = 3
    
    def __set_default_user_inputs(self):
            self.user_inputs = {
                "DIRECTORIES": {
                    "raw_patient_data": None,
                    "preprocessed_patient_data": None,
                    "patient_info": None,
                    "data_split": None,
                    "model_weights": None,
                    "model_inference": None
                },
                "TYPE_OF_TARGET_VOLUME": "ctv",
                "PARALLELIZATION": {
                    "number_of_processors": 1,
                },
                "DATA_PREPROCESSING": {
                    "oars": {
                        "contour_set": "clinical",
                    }
                }
            }
            self.patient_data_directory = None
            self.__logger.warning("No user inputs were provided. Setting default values.")
            
    def reset(self):
        
        self.patient_id = None
        self.original_ct_coordinates = None
        self.original_dose_coordinates = None
        self.mask_interpolation_technique = 'nearest'
        self.mask_generation_method = 'interpolate'
        self.write_new_hdf5_file = True
        self.relevant_masks = None
        self.compression = 'lzf'
        self.expected_data = ['ct', 'rtdose', 'rtplan', 'rtstruct']
        self.radiation_type = None
        self.echo_progress = True
        self.uniform_slice_thickness = True
        self.echo_level = 0
        self.coordinate_precision = 3
        self.equalize_dose_grid_dimensions = True
          
    def identify_patient_files(self, patient_data_directory = None, echo=False):
        """Function to identify the number of patient folders with all DICOM-RT files 
           needed for proper functioning of the code.

            Parameters
            ----------

            `patient_data_directory` : str
                The location of the patient data folders containing the required DICOM-RT files.

            `echo` : bool
                Flag to prompt funtion to write the number of patient folders found

            Returns
            -------

            `list`
                Names of folders for the patients found.
        """

        # check if a directory was specified
        if patient_data_directory is not None: self.patient_folders_directory = patient_data_directory

        # get a list of all of the folders in the directory
        folders = os.listdir(self.patient_data_directory)

        # check folder content to avoid future errors
        patient_files_info = {f:{'modalities':[], 'folder_directory':''} for f in folders}
        
        self.__logger.info(f'Checking the content of {self.patient_data_directory} to identify patient folders with the required DICOM-RT files.')
        
        for folder in folders:

            patient_folder_directory = os.path.join(self.patient_data_directory, folder)
            
            for root, _, files in os.walk(patient_folder_directory):
                
                for file in files:
        
                    # grab modality for DICOM file
                    file_directory = os.path.join(root, file)
                    
                    try:
                        ds = dcmread(file_directory)

                        modality = ds.data_element('Modality').value
                                                
                        patient_files_info[folder]['modalities'].append(modality.lower())
                                                
                    except:
                        self.__logger.warning(f"The content for the folder '{folder}' could not be read")
                        break
            
            # remove repeated modality values and sort the resulting list                                        
            patient_files_info[folder]['modalities'] = sorted(list(set(patient_files_info[folder]['modalities'])))
            patient_files_info[folder]['folder_directory'] = patient_folder_directory
         
        # Record the data folders with the required DICOM-RT files
        patients = []
        for p in patient_files_info.keys():
            if not all(m in patient_files_info[p]['modalities'] for m in self.expected_data):
                self.__logger.warning(f"The folder '{p}' is missing one or more of the required DICOM-RT files. "
                                      f"Current modalities: {', '.join(patient_files_info[p]['modalities'])}")                           
            else:
                patients.append(p)
        
        patients.sort(key=natural_keys)
        
        if echo: self.__logger.info(f'Found {len(patients)} patient folders in {patient_data_directory} with the required DICOM-RT files.')

        return patients
    
    def get_header_info(self, patient_id, file_type, save_to_file=False, echo=False):
          
        patient_files = self.run_initial_check(patient_id)
        
        if file_type == 'ct':
            files = patient_files['ct']
        elif file_type == 'dose':
            files = patient_files['dose']
        elif file_type == 'plan':
            files = patient_files['plan']
        elif file_type == 'structures':
            files = patient_files['structures']
        else:
            self.__logger.error(f'Invalid file type {file_type}.')
            return
        
        for n, f in enumerate(files):
            ds = dcmread(f)    
            
            if echo: print(ds) # print the header information
            
            # save pretty json to file in log directory
            if save_to_file:
                header_output_dir = os.path.join('logs', f'{file_type}_header_info_{file_type}_{n}.txt')
                with open(header_output_dir, 'w') as outfile:
                    print(ds, file=outfile)
            
            # if CT, exit
            if file_type == 'ct': break
                        
    def identify_radiation_type(self, patient_files, patient_id):
        """Function to identify the main type of radiation therapy used for a patient. The 
        function determines the most common radiation type used for the beams in the plan file.
        
        """
        
        if self.radiation_type is not None: return
        if 'rtdose' not in self.expected_data: return
                        
        try:      
            ds = dcmread(patient_files['plan'][0])      
            radiation_types = [b.RadiationType.lower() for b in ds.BeamSequence]
            unique_radiation_types = list(set(radiation_types))
            self.radiation_type = max(unique_radiation_types, key=radiation_types.count) 
        except:
            try:
                radiation_types = [b.RadiationType.lower() for b in ds.IonBeamSequence]
                unique_radiation_types = list(set(radiation_types))
                self.radiation_type = max(unique_radiation_types, key=radiation_types.count)                
            except:
                self.__logger.error(f'Failed to identify the radiation type for pat-{patient_id}.')
                self.__logger.info('If you know the radiation type, please specify it with the class attribute "radiation_type".')   
                                                        
    def run_initial_check(self, patient_id=None):
        
        assert self.patient_data_directory is not None
        if patient_id is not None:
            if type(patient_id) != type(""): patient_id = str(patient_id)
            self.patient_id = patient_id
        
        # prepare patient data directory
        patient_directory = os.path.join(self.patient_data_directory, self.patient_id)
        
        # Detect all files in the directory
        try:
            files = os.listdir(patient_directory)
        
            # Find directory of all type of DICOM files 
            patient_files = {'ct':[], 'plan':[], 'structures':[], 'dose':[]}
            
            # Discover all of the files for the patients
            for root, _, files in os.walk(patient_directory):
                for f in files:
                    # Get modality for the file
                    ds = dcmread(os.path.join(root,f))
                    modality = ds.data_element('Modality').value.lower()
                    
                    # Add file to the corresponding list
                    if modality == 'rtdose':
                        patient_files['dose'].append(os.path.join(root,f))
        
                    elif modality == 'rtplan':
                        patient_files['plan'].append(os.path.join(root,f))
                        
                    elif modality == 'rtstruct':
                        patient_files['structures'].append(os.path.join(root,f)) 
                        patient_files['structures'] = sorted(patient_files['structures'])
                        
                    elif modality == 'ct':
                        patient_files['ct'].append(os.path.join(root,f))     
                        
        except Exception as e:
            self.__logger.error(f"An error occured while trying to read the files for patient {self.patient_id}.")
            self.__logger.error(traceback.format_exc())
            return None
        
        # identify the radiation type
        if self.radiation_type is None: 
            try:
                self.identify_radiation_type(patient_files, self.patient_id)
            except:
                self.__logger.error(f'Failed to identify the radiation type for pat-{self.patient_id}.')
                self.radiation_type = 'not-specified'
    
        # Detect incomplete data
        file_types_dict = {'ct':'ct', 'rtdose':'dose', 'rtplan':'plan', 'rtstruct':'structures'}
        if any([patient_files[k]==[] for k in [file_types_dict[x] for x in self.expected_data]]):
            self.__logger.error(f'The full DICOM-RT set ({", ".join(self.expected_data)}) for patient {self.patient_id} could not be read.')
            self.__logger.info("Please change the expected data types by specifying the class attribute 'expected_data' or check the patient folder.")
            return None
        
        # Check the dose files to check for beam-specific or cumulative dose        
        patient_files['dose'] = self.__check_dose_files(patient_files['dose'])
            
        return patient_files
    
    def __check_dose_files(self, dose_files):
                
        dose_file_info = {n:{} for n in dose_files}
        
        for f in dose_files:
            with dcmread(f) as ds:
                dose_file_info[f]['dose_summation_type'] = ds.DoseSummationType.lower()
                dose_file_info[f]['data'] = ds.pixel_array * ds.DoseGridScaling
                try:
                    dose_file_info[f]['beam_number'] = int(ds.ReferencedRTPlanSequence[0][('300c','0020')][0][('300c','0004')][0][('300c','0006')].value)
                except:
                    dose_file_info[f]['beam_number'] = 'not_specified'
        
        # check the dose summation type
        if len(set([dose_file_info[f]['dose_summation_type'] for f in dose_files])) > 1:
            self.__logger.info(f'Multiple dose summation types were identified for patient {self.patient_id}: {", ".join([dose_file_info[f]["dose_summation_type"] for f in dose_files])}')

            # check total dose 
            cum_dose = np.sum([dose_file_info[f]['data'] for f in dose_files if dose_file_info[f]['dose_summation_type'] == 'beam'], axis=0)
            # grab the dose data for file with plan as dose summation type
            plan_dose = [dose_file_info[f]['data'] for f in dose_files if dose_file_info[f]['dose_summation_type'] == 'plan'][0]
            
            if np.round(np.abs(np.subtract(cum_dose,plan_dose).max())) > 0.0:
                self.__logger.error(f'The sum of the beam dose for patient {self.patient_id} does not match the dose for the plan file.')
                raise ValueError()
            else:
                return [f for f in dose_files if dose_file_info[f]['dose_summation_type'] == 'beam']
            
        else:
            # return the dose files
            return dose_files
                        
    def parse_dicom_files(self, patient_id=None, mask_names_only=False, mask_resolution = None, patient_data_directory = None):
        
        if patient_data_directory is not None: 
            self.user_inputs['DIRECTORIES']['raw_patient_data'] = patient_data_directory
        elif self.user_inputs['DIRECTORIES']['raw_patient_data'] is not None:
            patient_data_directory = self.user_inputs['DIRECTORIES']['raw_patient_data'] 
        else:
            self.__logger.error('No patient data directory was specified. Provide a directory or update the user inputs.')
            raise FileExistsError('Missing patient data directory.')
    
        # initial checks
        if patient_id is not None: 
            if type(patient_id) != type(''): patient_id = str(patient_id)
            self.patient_id = patient_id
        if patient_id is None and self.patient_id is None:
            self.__logger.error('Calling DICOM parsing function without specifying a patient ID. Please provide a patient ID or update the class attribute "patient_id".')
            raise Exception('Missing patient ID for DICOM parsing.')
        self.mask_resolution = mask_resolution if mask_resolution is not None else 'dose'
        
        if patient_data_directory  is not None and patient_data_directory .split('.')[-1] in ['h5', 'hdf5']:
            self.read_data_from_hdf5()
            return
        
        # Identify the file types in the patient folder and type of radiation therapy
        self.dicom_files = self.run_initial_check(self.patient_id)
                        
        # Parse the CT volume
        self.ct = self.parse_ct_study_files(self.dicom_files['ct'])

        # Parse the dose volume and (optionally) the plan 
        if 'plan' in self.dicom_files.keys() and self.dicom_files['plan'] != [] and self.dicom_files['dose'] != []:
            self.dose, self.plan = self.parse_rt_dose_files(self.dicom_files['dose'], self.dicom_files['plan'])
        elif self.dicom_files['dose'] != []:
            self.__logger.warning(f'No plan file was found for patient {self.patient_id}. Using default values for the plan.')
            self.dose = self.parse_rt_dose_files(self.dicom_files['dose'])
            self.plan = Plan()
                                     
        # Parse the contours
        self.contours = self.parse_structure_files(sorted(self.dicom_files['structures']), names_only=mask_names_only, resolution=self.mask_resolution)

    def parse_ct_study_files(self, files=None, patient_id = None, units='hu'):
      
        if patient_id is not None: 
            self.patient_id = str(patient_id)
            files = self.run_initial_check(self.patient_id)['ct']
            
        # Prepare CT volume
        ct_slices = {dcmread(f).ImagePositionPatient[-1]:dcmread(f).pixel_array for f in files}
        ## Construct the z coordinate array 
        z = sorted(list(ct_slices.keys()))
        ## Build 3D CT dataset
        data = np.array([ct_slices[i].astype(float) for i in z])
        ## Determine the number of slice spacings used for the CT data         
        z_spacing = list(set(list(np.round(np.array(z[1:]) - np.array(z[0:-1]), self.coordinate_precision))))
        z = np.round(z, self.coordinate_precision)
        
        if len(z_spacing) > 1:
            self.__logger.warning(f"Multiple slice thicknesses were identified for the CT data of patient {self.patient_id}: {', '.join([str(i) for i in z_spacing])} mm") 
        
        with dcmread(files[0]) as ds:
            
            # Grab the position of the patient
            patient_position = ds.PatientPosition.lower()
                   
            ## Grab image position (patient) attribute
            image_position = [np.round(p, self.coordinate_precision) for p in ds.ImagePositionPatient]
                       
            ## Update z-value of image position
            image_position[-1] = z[0]

            ## Store CT Study information
            xy_resolution = [np.round(float(i),self.coordinate_precision) for i in ds.PixelSpacing]
            rescale_slope = ds.RescaleSlope
            rescale_intercept = ds.RescaleIntercept     
            slice_thickness = np.round(ds.SliceThickness, self.coordinate_precision)
            
            # if np.round(ds.SliceThickness,self.coordinate_precision) not in z_spacing:
            #     self.__logger.warning(f'Mistmatch between the slice thickness {ds.SliceThickness}-mm and '
            #                           f'coordinate spacings ({", ".join([str(i) for i in z_spacing])})-mm for patient {self.patient_id}.')
                 
        ## Prepare the x and y coordinates
        x = np.round(np.arange(data.shape[2]) * xy_resolution[0] + image_position[0], self.coordinate_precision)
        y = np.round(np.arange(data.shape[1]) * xy_resolution[1] + image_position[1], self.coordinate_precision)
        dx, dy = xy_resolution
                 
        ## Interpolate volume if multiple slice thicknesses were used
        if len(z_spacing) > 1 and self.uniform_slice_thickness:
            min_dz = np.round(min(z_spacing), self.coordinate_precision)
            min_z, max_z = np.round(np.min(z), self.coordinate_precision), np.round(np.max(z), self.coordinate_precision)
            self.__logger.warning(f"Multiple slice thicknesses were identified for the CT data of patient {self.patient_id}: {', '.join([str(i) for i in z_spacing])} mm")  
            self.__logger.info(f'Interpolating CT data to achieve a uniform slice thickness of {min_dz}-mm')
            ### define new z-coordinates
            z_new = np.round(np.arange(min_z, max_z, min_dz), self.coordinate_precision)
            original_coordinates = (z, y, x)
            interpolation_coordinates = (z_new, y, x) 
            data = interpolate_volume(data, original_coordinates, interpolation_coordinates, 
                                      intMethod='linear', boundError=False, fillValue=0)
            z = z_new
            dz = np.round(min(z_spacing), self.coordinate_precision)
            
        else: 
            dz = np.array(z_spacing)
        
        ## Create coordinate object for CT data
        coordinates = Coordinates(x,y,z,dx,dy,dz,image_position)

        ## Convert units to HU if specified
        if units == 'hu': data = data * rescale_slope + rescale_intercept
        if units != 'hu': units = 'original'    

        ## Save a copy of the original CT information to help create the masks
        self.original_ct_coordinates = Coordinates(x,y,z,dx,dy,dz,image_position)
        self.original_ct_shape = data.shape

        return CT(data.shape, (dx,dy,dz), np.max(data), np.min(data), units, rescale_slope,
                  rescale_intercept, patient_position, data, slice_thickness, coordinates)
    
    def get_plan_info(self, dsP):
                    
        plan = Plan()

        plan.geometry = dsP.RTPlanGeometry.lower() if hasattr(dsP, "RTPlanGeometry") else 'not_specified'
        plan.patient_sex = dsP.PatientSex.lower() if dsP.PatientSex != '' else 'not_specified'
        plan.radiation_type = self.radiation_type.lower()
        
        for fgs in dsP.FractionGroupSequence:
            plan.number_of_fractions_planned = int(fgs.NumberOfFractionsPlanned) if hasattr(fgs, "NumberOfFractionsPlanned") else 0
            plan.number_of_beams = int(fgs.NumberOfBeams)
            dose_per_beam = []
            for rbs in fgs.ReferencedBeamSequence:
                dose_per_beam.append(float(rbs.BeamDose) if hasattr(rbs, "BeamDose") else 0.0)
                
            plan.dose_per_fraction = np.sum(dose_per_beam) 

            if hasattr(fgs, "DoseReferenceSequence"):
                for drs in fgs.DoseReferenceSequence:
                    if hasattr(drs, "DoseReferenceStructureType") and drs.DoseReferenceStructureType.lower() == 'site':
                        plan.dose_reference_type = drs.DoseReferenceType.lower() if hasattr(drs, "DoseReferenceType") else 'not_specified'
                        plan.dose_reference_description = drs.DoseReferenceDescription.lower() if hasattr(drs, "DoseReferenceDescription") else 'not_specified'
                        plan.dose_reference_dose = drs.TargetPrescriptionDose.lower() if hasattr(drs, "TargetPrescriptionDose") else 0.0
            
            plan.plan_label = dsP.RTPlanLabel.lower() if hasattr(dsP, "RTPlanLabel") else 'not_specified'

            patient_position = list(set([x.PatientPosition for x in dsP.PatientSetupSequence]))
            if len(patient_position) > 1:
                self.__logger.warning(f'Multiple patient positions were identified for patient {self.patient_id}: {", ".join(patient_position)}')
            else:
                plan.patient_position = patient_position[0].lower()
        
        information_sequence = dsP.IonBeamSequence if self.radiation_type == 'proton' else dsP.BeamSequence
        
        for b in information_sequence:
            cps = b.IonControlPointSequence[0] if self.radiation_type == 'proton' else b.ControlPointSequence[0]
            if b.TreatmentDeliveryType.lower() == 'setup': continue # skip setup beams
            
            plan.beam[int(b.BeamNumber)] = Beam()
            plan.beam[int(b.BeamNumber)].type = b.BeamType.lower()
            if hasattr(b, "VirtualSourceAxisDistances"):
                plan.beam[int(b.BeamNumber)].vsad = b.VirtualSourceAxisDistances
            else: 
                plan.beam[int(b.BeamNumber)].sad = float(b.SourceAxisDistance)
                
            plan.beam[int(b.BeamNumber)].gantry_angle = cps.GantryAngle
            plan.beam[int(b.BeamNumber)].patient_support_angle = cps.PatientSupportAngle
            plan.beam[int(b.BeamNumber)].isocenter = cps.IsocenterPosition
            plan.beam[int(b.BeamNumber)].treatment_delivery_type = b.TreatmentDeliveryType.lower()
            plan.beam[int(b.BeamNumber)].treatment_machine = b.TreatmentMachineName.lower()
                    
        return plan
    
    def get_additional_details_from_plan_file(self, dsP, dose, bn):
        
        # TODO: VMAT plans have more than one angle. This needs to be handled.
                
        # grab the beam dose      
        for fgs in dsP.FractionGroupSequence:
            for rbs in fgs.ReferencedBeamSequence:
                if int(rbs.ReferencedBeamNumber) == bn: 
                    dose.beam_dose = rbs.BeamDose if hasattr(rbs, "BeamDose") else 0
    
        information_sequence = dsP.IonBeamSequence if self.radiation_type == 'proton' else dsP.BeamSequence
 
        for b in information_sequence:
            if int(b.BeamNumber) == bn:
                
                cps = b.IonControlPointSequence[0] if self.radiation_type == 'proton' else b.ControlPointSequence[0]
                
                dose.beam_type = b.BeamType.lower()
                dose.radiation_type = b.RadiationType.lower()
                dose.beam_name = b.BeamName.lower()
                dose.beam_number = int(b.BeamNumber)
                dose.beam_description = b.BeamDescription.lower() if hasattr(b, "BeamDescription") else 'not_specified'
                dose.treatment_machine = b.TreatmentMachineName.lower()
                dose.final_cumulative_meterset_weight = b.FinalCumulativeMetersetWeight
                dose.scan_mode = b.ScanMode.lower() if hasattr(b, "ScanMode") else 'not_specified'
                dose.treatment_delivery_type = b.TreatmentDeliveryType.lower()
                dose.primary_dosimetric_units = b.PrimaryDosimeterUnit.lower()
                dose.number_of_control_points = int(b.NumberOfControlPoints)
                dose.gantry_angle = cps.GantryAngle
                dose.patient_support_angle = cps.PatientSupportAngle if hasattr(cps, "PatientSupportAngle") else 0.0
                dose.table_top_pitch_angle = cps.TableTopPitchAngle if hasattr(cps, "TableTopPitchAngle") else 0.0
                dose.table_top_roll_angle = cps.TableTopRollAngle if hasattr(cps, "TableTopRollAngle") else 0.0
                dose.isocenter = cps.IsocenterPosition
                
                if hasattr(b, "VirtualSourceAxisDistances"):
                    dose.vsad = b.VirtualSourceAxisDistances
                else:
                    dose.sad = b.SourceAxisDistance
                    dose.gantry_rotation_direction = b.GantryRotationDirection.lower() if hasattr(b, "GantryRotationDirection") else 'not_specified'
            
        return dose

    def parse_rt_dose_files(self, dose_files=None, plan_file=None, patient_id = None):
        
        if patient_id is not None: 
            self.patient_id = str(patient_id)
            dose_files = self.run_initial_check(self.patient_id)['dose']
            plan_file = self.run_initial_check(self.patient_id)['plan']

        assert dose_files is not None or patient_id is not None 
                        
        dose, args = {}, []
        for f in dose_files:
            with dcmread(f) as ds:
                                
                if len(dose_files) == 1: # handles the case for just one dose file
                    try: # check if the dose file is a beam-specific dose file
                        bn = int(ds.ReferencedRTPlanSequence[0][('300c','0020')][0][('300c','0004')][0][('300c','0006')].value)
                    except: # if not, assume it is a cumulative dose file
                        bn = 1
                    self.__logger.info(f'Only one dose file was found for patient {self.patient_id}.')
                    self.__logger.info('Assuming that the dose file contains the cummulative dose for the plan.')
                elif ds.DoseSummationType.lower() != 'plan': # handles the case for multiple dose files (beam-specific)
                    bn = int(ds.ReferencedRTPlanSequence[0][('300c','0020')][0][('300c','0004')][0][('300c','0006')].value)
                       
                # Grab data
                data = ds.pixel_array * ds.DoseGridScaling
                # Grab some data properties
                units = ds.DoseUnits
                xy_resolution = [np.round(float(x), self.coordinate_precision) for x in ds.PixelSpacing]
                dose_grid_scaling = float(ds.DoseGridScaling)
                image_position = [np.round(float(i), self.coordinate_precision) for i in ds.ImagePositionPatient]
                grid_offset_vector = np.round(np.array(ds.GridFrameOffsetVector), self.coordinate_precision)
                
                # Prepare coordinates
                x = np.round(np.arange(ds.Columns)*xy_resolution[0] + image_position[0], self.coordinate_precision)
                y = np.round(np.arange(ds.Rows)*xy_resolution[1] + image_position[1], self.coordinate_precision)
                z = np.round(grid_offset_vector+ image_position[2], self.coordinate_precision)
                ## Determine the number of slice spacings used for the dose data         
                z_spacing = list(set(list(np.round(np.array(z[1:]) - np.array(z[0:-1]), self.coordinate_precision))))
                ## Interpolate volume if multiple slice thicknesses were used
                if len(z_spacing) > 1:
                    min_dz = min(z_spacing)
                    min_z, max_z = np.round(np.min(z), self.coordinate_precision), np.round(np.max(z), self.coordinate_precision)
                    self.__logger.warning(f'Two or more slice thicknesses identified for the dose data of patient {self.patient_id}.')    
                    self.__logger.info(f'Interpolating dose data to achieve a uniform slice thickness of {min_dz}-mm')
                    ### define new z-coordinates
                    z_new = np.round(np.arange(min_z, max_z, min_dz), self.coordinate_precision)
                    original_coordinates = (z, y, x)
                    interpolation_coordinates = (z_new, y, x) 
                    data = interpolate_volume(data, original_coordinates, interpolation_coordinates, 
                                              intMethod='linear', boundError=False, fillValue=0)
                    z = z_new
                
                # Grab coordinate information
                dx, dy, dz = xy_resolution + [min(z_spacing)]
                coordinates = Coordinates(x,y,z,dx,dy,dz,image_position)
                
                # Prepare dose object
                if self.radiation_type == 'photon':
                    dose[bn] = PhotonDose(*[data.shape, data.max(), data.min(), (dx,dy,dz), 
                                            dose_grid_scaling, units, data, coordinates])
                else:
                    dose[bn] = ProtonDose(*[data.shape, data.max(), data.min(), (dx,dy,dz), 
                                            dose_grid_scaling, units, data, coordinates])
                
                # Grab additional information from plan file if available
                if plan_file is not None and plan_file != []:
                    with dcmread(plan_file[0]) as dsP:                                         
                        dose[bn] = self.get_additional_details_from_plan_file(dsP, dose[bn], bn)
                                            
        if len(set([dose[bn].data.shape for bn in dose.keys()])) > 1:      
            self.__logger.warning(f'Not all of the dose volumes have the same shape for patient {self.patient_id}.')
            if self.equalize_dose_grid_dimensions:
                self.__logger.info('Equalizing dose grid dimensions')
                dose = self.__equalize_dose_grid_dimensions(dose)

        # Save a copy of the original dose information to help create the masks
        assert len(set([dose[bn].data.shape for bn in dose.keys()])) == 1    
        self.original_dose_coordinates = copy.deepcopy(dose[bn].coordinates)
        self.original_dose_shape = data.shape

        # Grab additional information from plan file if available
        if plan_file is not None and plan_file != []: 
            with dcmread(plan_file[0]) as dsP:
            
                plan = self.get_plan_info(dsP)

            return dose, plan
        else:
            return dose
        
    def __equalize_dose_grid_dimensions(self, dose):
            
        max_shape = np.max([dose[bn].data.shape for bn in dose.keys()], axis=0)

        bn_to_correct = []

        for k in dose.keys():
            num_max_dims = []
            for n,s in enumerate(dose[k].data.shape):
                if s == max_shape[n]: 
                    num_max_dims.append(k)
            if len(num_max_dims) == 3: 
                max_dim_bn = k
                break

        if 'max_dim_bn' not in locals(): raise ValueError('Unable to find a beam with the maximum shape along all dimensions.')
        
        # determine patient(s) needing correction
        for k in dose.keys():
            for n,s in enumerate(dose[k].data.shape):
                if s != max_shape[n]: 
                    bn_to_correct.append(k)
                    
        for b in bn_to_correct:
            original_coordinates = (dose[b].coordinates.z, dose[b].coordinates.y, dose[b].coordinates.x) 
            interpolation_coordinates = (dose[max_dim_bn].coordinates.z, dose[max_dim_bn].coordinates.y, dose[max_dim_bn].coordinates.x)   
            data = interpolate_volume(dose[b].data, original_coordinates, interpolation_coordinates,
                                    intMethod='linear', boundError=0, fillValue=0)
            
            dose[b].data = data
            dose[b].coordinates = copy.deepcopy(dose[max_dim_bn].coordinates)
            dose[b].shape = data.shape
                        
        return dose
                       
    def parse_structure_files(self, files = None, patient_id = None, mask_names = None, resolution = 'dose', names_only = False):
                
        # Fetch structure files if they are not provided
        if files is None:
            self.patient_id = patient_id if patient_id is not None else self.patient_id
            assert self.patient_id is not None, 'Patient ID is missing.'
            self.structure_files = self.run_initial_check(self.patient_id)['structures']
        else:
            self.structure_files = files
            
        # ensure that mask names are in a list
        if type(mask_names) != type([]) and type(mask_names) == type(''): mask_names = [mask_names]
        
        # Find the name of all of the masks in the plan
        self.all_mask_names = []
        for s_file in self.structure_files:                       
            with dcmread(s_file) as ds:
                self.all_mask_names += list(self.read_structure(ds).keys())
        
        # return all available masks names only: avoids building a volume
        if names_only: return self.all_mask_names

        # grab ct information if the CT study has not been read yet
        if self.original_ct_coordinates is None:
            self.parse_ct_study_files(patient_id=self.patient_id)

        if self.original_dose_coordinates is None and resolution.lower() == 'dose':
            self.parse_rt_dose_files(patient_id = self.patient_id)

        # Grab the coordinates for the contours
        coordinates = copy.deepcopy(self.original_ct_coordinates) if resolution.lower() == 'ct' else copy.deepcopy(self.original_dose_coordinates)
    
        # Ensure that only the relevant masks for the patient are parsed
        if self.relevant_masks is not None and mask_names is None:
            assert all([m in self.all_mask_names for m in self.relevant_masks]), 'One or more of the specified masks is not available for the patient.'
            self.all_mask_names = self.relevant_masks
                   
        # prepare the output dictionary
        contours = {}
        
        for s_file in self.structure_files:
                                
            with dcmread(s_file) as ds: 
                                    
                structures = self.read_structure(ds)   
                
                for k in structures.keys() if mask_names is None else mask_names:
                                            
                    # grab the data for the mask
                    if k not in self.all_mask_names:
                            self.__logger.warning(f'A mask "{k}" was not found for patient-{self.patient_id}.')
                            data = self.__return_empty_mask(resolution)   
                    elif k in self.all_mask_names and k not in structures.keys():
                        continue
                    else:
                        data = self.get_mask(structures, k, resolution = resolution)
                    
                    # generate a unique name for the mask
                    name = self.__get_unique_mask_name(k, contours.keys())
                    
                    # create the mask object
                    try:
                        contours[name] = Mask(data, resolution, coordinates, 
                                            structures[k]['number'], 
                                            structures[k]['color'],
                                            structures[k]['generation_algorithm'])
                    except:
                        if self.echo_level > 0:
                            self.__logger.error(f'Failed to build the mask "{k}" for pat-{self.patient_id}.')
                            self.__logger.error(traceback.format_exc())
                            
                        contours[name] = Mask(self.__return_empty_mask(resolution), resolution, coordinates, 
                                                'not_specified', 'not_specified', 'not_specified')

        return contours 
    
    def __get_unique_mask_name(self, mask_name, current_mask_names):   
        if mask_name not in current_mask_names: return mask_name 
        mask_number = 1
        final_name = f'{mask_name}_{mask_number}'
        while final_name in current_mask_names:
            mask_number += 1
            final_name = f'{mask_name}_{mask_number}'
        return final_name
    
    def __return_empty_mask(self, resolution):
        try:
            if 'ct' in resolution:
                return np.zeros_like(self.ct.data)
            else:
                bn = list(self.dose.keys())[0]
                return np.zeros_like(self.dose[bn].data)
        except:
            return 0

    def read_structure(self, ds):
        """Auxiliary function for reading the content of the structure file.

        Parameters
        ----------
        ds : pydycom object
            Handle for the file opened using the pydicom module 

        Returns
        -------
        dict
            Dictionary contraining the controu data, color, number, and name of 
            the contrours.
        """
        contours = {}
         
        for i in range(len(ds.ROIContourSequence)):
            contour = {}
            try:
                contour['contour_data'] = [s.ContourData for s in ds.ROIContourSequence[i].ContourSequence]
                contour['color'] = ds.ROIContourSequence[i].ROIDisplayColor
                contour['number'] = ds.ROIContourSequence[i].ReferencedROINumber
                contour['generation_algorithm'] = str(ds.StructureSetROISequence[i].ROIGenerationAlgorithm).lower()
                contours[str(ds.StructureSetROISequence[i].ROIName).lower()] = contour
            except:
                pass
            
        return contours

    def get_mask(self, contours, name, resolution = 'ct', method = 'interpolate'):
        """Function to construct a binary mask using the set of contouring coordinate\n 
        triplets stored in the structure dicom file (RTS).
                   
        Created by Ivan Vazquez in collaboration with Ming Yang
        
        Last Update: Fall 2023

        Parameters
        ----------
            `contours (list)`: List containing the contouring data (coordinates), color, number, and name of the contours.
            `name` (str): Name of specific contour to build into a mask. Defaults to None.
            `resolution` (str, optional): Resolution of the grid used to construct the mask. It can be 'ct' or 'dose'. Defaults to 'ct'.
            `method` (str, optional): Method used to produce volumes at the resolution of the dose grid. Defaults to 'interpolate'.

        Returns
        -------
            `ndarray(dtype=float, ndim=2)`: Binary mask 
        """   
          
        method = self.mask_generation_method if method is None else method
           
        z = self.original_ct_coordinates.z[:] if method == 'interpolate' else self.original_dose_coordinates.z[:]
        y_0 = self.original_ct_coordinates.image_position[1] if method == 'interpolate' else self.original_dose_coordinates.image_position[1]
        dy = self.original_ct_coordinates.dy if method == 'interpolate' else self.original_dose_coordinates.dy
        x_0 = self.original_ct_coordinates.image_position[0] if method == 'interpolate' else self.original_dose_coordinates.image_position[0]
        dx = self.original_ct_coordinates.dx if method == 'interpolate' else self.original_dose_coordinates.dx
        z_min, z_max = np.round(z.min(),self.coordinate_precision), np.round(z.max(),self.coordinate_precision)
            
        # allocate volume for mask using the shape of the original CT volume
        shape = self.original_dose_shape if resolution == 'dose' and method != 'interpolate' else self.original_ct_shape
        mask = np.zeros(shape, dtype=np.uint8) 
        
        # round the z coordinates to avoid floating point errors
        z = np.round(z,self.coordinate_precision)
                  
        ## Grab the contour data that matches the name of the desired mask
        contour_data = [np.array(i).reshape(-1,3) for i in [contours[c]['contour_data'] for c in contours.keys() if c.lower() == name.lower()][0]]
                    
        for nodes in contour_data: 
            
            z_node = np.round(nodes[0, 2],self.coordinate_precision)  
                                 
            if np.logical_and(z_node >= z_min, z_node <= z_max): # ignore slices outside of the CT volume
            
                z_index = np.where(z == z_node)[0][0]
         
                r = (nodes[:, 1] - y_0) / dy
                c = (nodes[:, 0] - x_0) / dx 
                
                # make values larger than max index equal to max index
                r[np.where(r > mask.shape[1]-1)] = mask.shape[1]-1
                c[np.where(c > mask.shape[2]-1)] = mask.shape[2]-1
                
                rr, cc = polygon(r, c)
            
                mask[z_index, rr, cc] += 1
          
        mask[np.where(mask>1)] = 0 # account for holes (mask ==2) in a contour
                      
        if resolution == 'dose' and method == 'interpolate':
            oc = (self.original_ct_coordinates.z, self.original_ct_coordinates.y, self.original_ct_coordinates.x)
            ic = (self.original_dose_coordinates.z, self.original_dose_coordinates.y, self.original_dose_coordinates.x)
            return interpolate_volume(mask, oc, ic, intMethod=self.mask_interpolation_technique, boundError=0, fillValue=0)
        
        return mask
        
    def convert_hu_to_rsp(self, ct=None, in_place=True, interpolation_kind = 'linear', lut_directory=None):
        
        # Read LUT for HU to RSP conversion
        if lut_directory is not None: self.rsp_lut_directory = lut_directory
        if self.rsp_lut_directory is None: 
            self.__logger.error('No LUT directory was provided for the HU to RSP conversion.')
            raise ValueError('No LUT directory was found or provided for the HU to RSP conversion.')
        
        rsp_lut_dir = os.path.join(self.rsp_lut_directory, 'mda_relative_stopping_power.csv') 
        
        try:
            HU_2_RSP_LUT = pandas.read_csv(rsp_lut_dir)
        except:
            self.__logger.error(f'Error reading the HU to RSP LUT from "{rsp_lut_dir}". This should be a CSV file.')
            raise ValueError(f'Error reading the HU to RSP LUT from "{rsp_lut_dir}".')
        HU = HU_2_RSP_LUT['HU'].values
        rsp = HU_2_RSP_LUT['rsp'].values
        
        # create interpolating function
        rsp2Hu = interp1d(HU, rsp, kind = interpolation_kind)

        if ct is None: ct = self.ct
        data = ct.data

        # Ensure that the min and max values of CT's HU values are inside acceptable range
        badLowInds = np.where(data < HU.min())
        data[badLowInds] = HU.min()

        badHighInds = np.where(data > HU.max())
        data[badHighInds] = HU.max()

        # Use interpolating function to convert the CT array
        rspVol = rsp2Hu(data.flatten()).reshape(ct.shape)

        if in_place: 
            ct.data = rspVol
        else:
            return rspVol

    def __get_patient_list(self, patient_data_directory):
        if os.path.isdir(patient_data_directory):
            return self.identify_patient_files(patient_data_directory)
        elif os.path.isfile(patient_data_directory) and patient_data_directory.split('.')[-1] in ['h5', 'hdf5']:
            with h5py.File(patient_data_directory, 'r') as hf:
                return list(hf.keys())
        else:
            raise Exception (f"Unable to work with the patient data directory: {patient_data_directory}")

    def get_dicom_data_report(self, patient_list=None, save = True):
        
        # grab list of patient files to explore
        if patient_list is None: 
            patient_list = self.__get_patient_list(self.patient_data_directory)
        else:
            if type(patient_list) != type([]): patient_list = [patient_list]
            patient_list = sorted([str(n) for n in patient_list])
        
        if self.parallelize is None: self.identify_parallel_capabilitie  # check if parallel processing should be used
        
        def get_report(my_patients, infer_rx_dose, process_id=None, save_file = True):
            
            data_report = {p:{} for p in my_patients}
             
            # check if parallel processing is possible
            if not self.parallelize:
                if self.echo_progress: my_patients = tqdm(my_patients, desc="Generating basic report", leave=True)
            
            for p in my_patients: 
                
                self.reset()
                                            
                try:        
                    self.parse_dicom_files(p, mask_names_only=True) # parse the data and grab contour names only
                    beams = list(self.plan.beam.keys())
                    data_report[p]['number_of_beams'] = len(beams) if beams != [] else 1
                    data_report[p]['radiation_type'] = self.radiation_type
                    data_report[p]['gantry_angles'] = [self.plan.beam[b].gantry_angle for b in beams]
                    data_report[p]['couch_angles'] = [self.plan.beam[b].patient_support_angle for b in beams]
                    data_report[p]['ct_array_dimensions'] = self.ct.data.shape
                    data_report[p]['dose_array_dimensions'] = self.dose[beams[0]].data.shape
                    data_report[p]['dose_array_resolution'] = {'dx':self.dose[beams[0]].coordinates.dx,
                                                               'dy':self.dose[beams[0]].coordinates.dy,
                                                               'dz':self.dose[beams[0]].coordinates.dz}
                    if self.radiation_type == 'proton':
                        data_report[p]['vsad'] = [self.plan.beam[b].vsad for b in beams]
                    else:
                        data_report[p]['sad'] = [self.plan.beam[b].sad for b in beams]
                        data_report[p]['gantry_rotation_direction'] = self.dose[beams[0]].gantry_rotation_direction
                    data_report[p]['isocenter'] = [self.plan.beam[b].isocenter for b in beams]
                    data_report[p]['contours'] = ','.join(sorted(self.contours))
                    data_report[p]['dose_reference_dose']= self.plan.dose_reference_dose
                    data_report[p]['dose_reference_type']= self.plan.dose_reference_type
                    data_report[p]['dose_reference_description']= self.plan.dose_reference_description
                    data_report[p]['beam_type'] = list(set([self.plan.beam[b].type for b in beams]))
                    data_report[p]['patient_position'] = self.ct.patient_position
                                       
                except Exception as e:
                    self.__logger.error(f"The data for pat-{p} could not be analyzed")
                    self.__logger.error(traceback.format_exc())
                    
            if save_file:
                tag = f'_{process_id}' if process_id is not None else ''
                with open(os.path.join('temp','data',f'basic_data_report{tag}.pickle'), 'wb') as handle:
                    pickle.dump(data_report, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        
            if not self.parallelize: return data_report
                
        if self.parallelize:
            
            self.__logger.info(f'Parallelizing the data report generation using {self.n_threads} threads.')
                        
            processes = [] # initialize a list to store the processes
                
            # divide the patients into groups based on the number of available threads
            chunked_patients = [x for x in np.array_split(patient_list, self.n_threads) if x.size != 0]
            
            # create a process for each thread
            for n, patients in enumerate(chunked_patients):
                p = mp.Process(target=get_report, args=(patients, infer_rx_dose, n, save,))
                p.start()
                processes.append(p)
            
            for p in processes:
                p.join()   
            
        else:
            data_report = get_report(patient_list, infer_rx_dose, save_file = save)
        
        # merge data reports and remove temp files
        if self.parallelize and save:
            data_report = {}
            for n in range(len(chunked_patients)):
                with open(os.path.join('temp','data',f'basic_data_report_{n}.pickle'), 'rb') as handle:
                    data_report.update(pickle.load(handle))
                os.remove(os.path.join('temp','data',f'basic_data_report_{n}.pickle'))
            # save merged data report
            with open(os.path.join('temp','data','basic_data_report.pickle'), 'wb') as handle:
                pickle.dump(data_report, handle, protocol=pickle.HIGHEST_PROTOCOL)
                       
        self.__logger.info(f"Finished generating basic report for all detected patients.")

        return data_report
    
    def infer_rx_dose(self, patient_list=None, min_infered_rx_dose = 20, get_dose_statistics = True):
            
        if patient_list is None: 
            patient_list = self.__get_patient_list(self.patient_data_directory)
        else:
            if type(patient_list) != type([]): patient_list = [patient_list]
            patient_list = sorted([str(n) for n in patient_list])
            
        target_properties = {p : {} for p in patient_list} # TODO: change this to dose information
        rx_dose_info ={'AID':[p for p in patient_list], 'Rx Dose':[], 'Plan Type':[], 'dose_scale':[], 'site':[]}
        
        if self.echo_progress: patient_list = tqdm(patient_list, desc="Infering prescription doses:", leave=True)

        for p in patient_list: 
                
            self.reset()
                                        
            self.parse_dicom_files(p, mask_names_only=True)
            
            # Indentify target volumes
            all_target_volumes = [x for x in self.contours if self.user_inputs["TYPE_OF_TARGET_VOLUME"] in x]
            len_target_name = len(self.user_inputs["TYPE_OF_TARGET_VOLUME"])
            target_volumes = [x for x in all_target_volumes if x[:len_target_name] == self.user_inputs["TYPE_OF_TARGET_VOLUME"]]        
            
            # Infer the prescription dose 
            rx = [self.__find_posible_rx_dose(x) for x in target_volumes]
            rx = list(set([str(x) for x in rx if float(x) >=min_infered_rx_dose]))    
            rx = [x for x in rx if x != '0']
            if len(rx) > 1: # if multiple rx doses are found, check if they are multiples of each other
                rx = [float(x)/100 if float(x) > 100 else float(x) for x in rx]
                rx = [str(x) for x in set(rx)]
                
            target_properties[p]['infered_rx_dose'] = ','.join(rx) if rx != [] else '0'
            target_properties[p]['all_target_like_structures'] = ','.join([x for x in self.contours if self.user_inputs["TYPE_OF_TARGET_VOLUME"] in x])
            rx_dose_info['Rx Dose'].append(','.join(rx) if rx != [] else '0')
            rx_dose_info['Plan Type'].append('Unknown')
            rx_dose_info['dose_scale'].append(1.0)
            rx_dose_info['site'].append('Unknown') 
            
            cum_dose = self.cumulative_dose
            
            for t in target_volumes:
                target_properties[p][t] = {} # TODO: use dose information variable instead of results
                self.__find_posible_rx_dose(t)
                # parse target mask
                target_data = self.parse_structure_files(mask_names = t, resolution = 'dose')[t].data
                # get the cumulative dose
                dose_in_mask = cum_dose[np.where(target_data>0)]
                
                # find possible rx dose
                rx_from_name = self.__find_posible_rx_dose(t)
                # find dose statistics
                target_properties[p][t]['D95'] = np.percentile(dose_in_mask, 100-max(0, min(100, 95)))
                target_properties[p][t]['D98'] = np.percentile(dose_in_mask, 100-max(0, min(100, 98)))
                target_properties[p][t]['D2'] = np.percentile(dose_in_mask, 100-max(0, min(100, 2)))
                target_properties[p][t]['mean_dose'] = np.mean(dose_in_mask)
                target_properties[p][t]['max_dose'] = np.max(dose_in_mask)
                
        # save target properties as json nicely formatted
        with open(os.path.join('temp','data','target_properties.json'), 'w') as f:
            json.dump(target_properties, f, indent=4)         
        
        # save rx dose information as csv
        rx_dose_info = pandas.DataFrame(rx_dose_info)
        rx_dose_info.to_csv(os.path.join('temp','data','inferred_ rx_dose_info.csv'), index=False)
        
    def __find_posible_rx_dose(self, string):
        string = string.replace(self.user_inputs["TYPE_OF_TARGET_VOLUME"], '')
        if 'mm' in string: return 0 # return 0 if the string contains units of mm
        pattern = r'\d+(?:\.\d+)?'  # Matches whole numbers and decimal numbers
        matches = {float(x):x for x in re.findall(pattern, string)}
        return matches[max(matches.keys())] if matches != {} else 0

    def store_patient_data_as_hdf5(self, patient_data_directory=None, patient_list=None, mask_resolution = None, output_directory=None):
        """Transfer data from one or more patients from DICOM-RT formats to HDF5 along with 
        all of the necessary details like coordinates, location of the isocenter, VSAD, and more. 
        This allows for a significant increase in I/O speed but increases the hard-drive memory consumption by 
        creating (in many cases) a large HDF5 file with all of the patient data. Please ensure that enough memory 
        is available when using this method of data wrangling.

        Parameters
        ----------
        patient_data_directory : str, optional
            Location of patient data folders. If None, the location in the `user_inputs.json` file will be used, by default None
        patient_list : list, optional
            List of patient IDs to transfer into the HDF5 file, by default None
        output_directory : str, optional
            Folder directory for the output HDf5 file, by default None
        """

        self.mask_resolution = mask_resolution if mask_resolution is not None else 'dose'

        if patient_data_directory is not None: self.patient_data_directory = patient_data_directory

        if output_directory is None: 
            output_directory = os.path.join('temp','data', 'patient_data.h5')   

        # Create h5file
        if self.write_new_hdf5_file:
            out_file = h5py.File(output_directory,'w'); out_file.close()

        if patient_list is None: patient_list = self.identify_patient_files()

        if self.echo_progress: patient_list = tqdm(patient_list, desc="Saving HDF5 patient file", leave=True)
        
        for p in patient_list:

            self.patient_id, files = p, self.run_initial_check(p)
            
            self.ct = self.parse_ct_study_files(files['ct'])

            self.append_data_to_hdf5_file(output_directory, data_type='ct')
            
            # Parse the dose volime and (optinally) the plan 
            if 'plan' in files.keys() and files['plan'] != []:
                self.dose, self.plan = self.parse_rt_dose_files(files['dose'], files['plan'])
            elif 'dose' in files.keys() and files['dose'] != []:
                self.dose = self.parse_rt_dose_files(files['dose'])
                self.plan = Plan()

            self.append_data_to_hdf5_file(output_directory, data_type='dose')

            # Parse the contours
            for f in files['structures']:
            
                self.contours = self.parse_structure_files([f], resolution=self.mask_resolution)

                self.append_data_to_hdf5_file(output_directory, 'contours')

            self.parse_dicom_files(p)

        self.__logger.info(f"The patient data was stored as an HDF5 file in {output_directory}.")
    
    def append_data_to_hdf5_file(self, output_directory, data_type):
        """Append the data for each patient to an HDF5 file for faster I/O

        Parameters
        ----------
        `output_directory` : str
            directory for the folder where the patient data file will be stored.
        """

        if data_type == 'ct':
            with h5py.File(output_directory,'a') as f:

                # CT data
                dset = f.create_dataset(name='/'.join([self.patient_id, 'ct']), data = self.ct.data, compression = self.compression)
                dset.attrs['units'] = self.ct.units
                dset.attrs['rescale_slope'] = self.ct.rescale_slope
                dset.attrs['rescale_intercept'] = self.ct.rescale_intercept
                dset.attrs['units'] = self.ct.units
                dset.attrs['resolution'] = self.ct.resolution
                dset.attrs['x'] = self.ct.coordinates.x
                dset.attrs['y'] = self.ct.coordinates.y
                dset.attrs['z'] = self.ct.coordinates.z
                dset.attrs['dx'] = self.ct.coordinates.dx
                dset.attrs['dy'] = self.ct.coordinates.dy
                dset.attrs['dz'] = self.ct.coordinates.dz
                dset.attrs['image_position'] = self.ct.coordinates.image_position

        elif data_type == 'dose':
                
            with h5py.File(output_directory,'a') as f:
                
                # Dose data
                if self.radiation_type.lower() == 'pronton':
                    for b in self.dose.keys():
                        dset = f.create_dataset(name='/'.join([self.patient_id, 'dose', str(b)]), data = self.dose[b].data, compression = self.compression )
                        dset.attrs['dose_grid_scaling'] = self.dose[b].dose_grid_scaling
                        dset.attrs['dose_units'] = self.dose[b].dose_units
                        dset.attrs['beam_type'] = self.dose[b].beam_type
                        dset.attrs['vsad'] = self.dose[b].vsad
                        dset.attrs['number_of_control_points'] = self.dose[b].number_of_control_points
                        dset.attrs['gantry_angle'] = self.dose[b].gantry_angle
                        dset.attrs['patient_support_angle'] = self.dose[b].patient_support_angle
                        dset.attrs['table_top_pitch_angle'] = self.dose[b].table_top_pitch_angle
                        dset.attrs['table_top_roll_angle'] = self.dose[b].table_top_roll_angle
                        dset.attrs['isocenter'] = self.dose[b].isocenter
                        dset.attrs['resolution'] = self.dose[b].resolution
                        dset.attrs['x'] = self.dose[b].coordinates.x
                        dset.attrs['y'] = self.dose[b].coordinates.y
                        dset.attrs['z'] = self.dose[b].coordinates.z
                        dset.attrs['dx'] = self.dose[b].coordinates.dx
                        dset.attrs['dy'] = self.dose[b].coordinates.dy
                        dset.attrs['dz'] = self.dose[b].coordinates.dz
                        dset.attrs['image_position'] = self.dose[b].coordinates.image_position
                        
                #TODO: add values for photon dose
                
                # Plan data
                plan = f.create_group(name=f'{self.patient_id}/plan')
                plan.attrs['number_of_beams'] = self.plan.number_of_beams
                plan.attrs['plan_label'] = self.plan.plan_label
                plan.attrs['patient_sex'] = self.plan.patient_sex
                plan.attrs['plan_name'] = self.plan.plan_name

        elif data_type == 'contours':
            with h5py.File(output_directory,'a') as f:
                for c in self.contours.keys():
                    name = str(c).replace("/","-")
                    dset = f.create_dataset(name='/'.join([self.patient_id, 'contours', self.contours[c].structure_set, str(name)]), 
                                            data = self.contours[c].data, compression = self.compression )
                    dset.attrs['resolution'] = self.contours[c].resolution

    def read_data_from_hdf5(self, patient_id = None, contours_type='clinical'):
        """Read patient data from HDF5 files generated by this class using  the `store_patient_data_as_hdf5` method. 
        This allows for fast I/O but increases the hard-drive memory consumption by creating (in many cases) a large 
        HDF5 file with all of the patient data. Please ensure that enough memory is available when using this 
        method of data wrangling.

        Parameters
        ----------
        `patient_id` : str, optional
            ID of the patient data to read, by default None
        """

        if patient_id is not None: self.patient_id = patient_id

        with h5py.File(self.user_inputs["DIRECTORIES"]["raw_patient_data"],'r') as f:

            # Plan
            plan = f[f'{self.patient_id}/plan']
            number_of_beams = plan.attrs['number_of_beams']
            patient_sex = plan.attrs['patient_sex']            
            plan_label = plan.attrs['plan_label'] 
            plan_name = plan.attrs['plan_name']
            modality = plan.attrs['modality']
            self.plan = Plan(number_of_beams, patient_sex, plan_label, plan_name, modality)

            # CT
            params = [f[f'{self.patient_id}/ct'][...].shape]
            params.append(f[f'{self.patient_id}/ct'].attrs['resolution'])
            params.append(f[f'{self.patient_id}/ct'][...].max())
            params.append(f[f'{self.patient_id}/ct'][...].min())
            params.append(f[f'{self.patient_id}/ct'].attrs['units'])
            params.append(f[f'{self.patient_id}/ct'].attrs['rescale_slope'])
            params.append(f[f'{self.patient_id}/ct'].attrs['rescale_intercept'])
            params.append(f[f'{self.patient_id}/ct'][...])
            x, y, z = f[f'{self.patient_id}/ct'].attrs['x'], f[f'{self.patient_id}/ct'].attrs['y'], f[f'{self.patient_id}/ct'].attrs['z']
            dx, dy, dz = f[f'{self.patient_id}/ct'].attrs['dx'], f[f'{self.patient_id}/ct'].attrs['dy'], f[f'{self.patient_id}/ct'].attrs['dz']
            image_position = f[f'{self.patient_id}/ct'].attrs['image_position']
            self.original_ct_coordinates = Coordinates(x,y,z,dx,dy,dz,image_position)
            params.append(self.original_ct_coordinates)
            self.ct = CT(*params)

            # Dose
            self.dose = {int(bn):None for bn in f[f'{self.patient_id}/dose'].keys()}
            for k in self.dose.keys():
                params = [f[f'{self.patient_id}/dose/{k}'][...].shape]
                params.append(f[f'{self.patient_id}/dose/{k}'][...].max())
                params.append(f[f'{self.patient_id}/dose/{k}'][...].min())
                params.append(f[f'{self.patient_id}/dose/{k}'].attrs['resolution'])
                params.append(f[f'{self.patient_id}/dose/{k}'].attrs['dose_grid_scaling'])
                params.append(f[f'{self.patient_id}/dose/{k}'].attrs['dose_units'])
                params.append(f[f'{self.patient_id}/dose/{k}'][...])
                x, y, z = f[f'{self.patient_id}/dose/{k}'].attrs['x'], f[f'{self.patient_id}/dose/{k}'].attrs['y'], f[f'{self.patient_id}/dose/{k}'].attrs['z']
                dx, dy, dz = f[f'{self.patient_id}/dose/{k}'].attrs['dx'], f[f'{self.patient_id}/dose/{k}'].attrs['dy'], f[f'{self.patient_id}/dose/{k}'].attrs['dz']
                image_position = f[f'{self.patient_id}/dose/{k}'].attrs['image_position']
                self.original_dose_coordinates = Coordinates(x,y,z,dx,dy,dz,image_position)
                params.append(self.original_dose_coordinates)
                params.append(f[f'{self.patient_id}/dose/{k}'].attrs['modality'])
                params.append(f[f'{self.patient_id}/dose/{k}'].attrs['beam_type'])
                params.append(f[f'{self.patient_id}/dose/{k}'].attrs['vsad'])
                params.append(f[f'{self.patient_id}/dose/{k}'].attrs['number_of_control_points'])
                params.append(f[f'{self.patient_id}/dose/{k}'].attrs['gantry_angle'])
                params.append(f[f'{self.patient_id}/dose/{k}'].attrs['patient_support_angle'])
                params.append(f[f'{self.patient_id}/dose/{k}'].attrs['table_top_pitch_angle'])
                params.append(f[f'{self.patient_id}/dose/{k}'].attrs['table_top_roll_angle'])
                params.append(f[f'{self.patient_id}/dose/{k}'].attrs['isocenter'])
                self.dose[k] = Dose(*params)
                
            #TODO: differentiate between photon and proton dose

            # Contours
            self.contours = {c:None for c in  f[f'{self.patient_id}/contours/{contours_type}'].keys()}
            for k in self.contours.keys():
                data = f[f'{self.patient_id}/contours/{contours_type}/{k}'][...]
                resolution = f[f'{self.patient_id}/contours/{contours_type}/{k}'].attrs['resolution']
                if resolution == 'ct':
                    coordinates = self.original_ct_coordinates
                else:
                    coordinates = self.original_dose_coordinates
                self.contours[k] = Mask(data, resolution, coordinates)
            # Grab CTVs in case the desired contours are automatic
            if contours_type == 'auto':
                for k in [c for c in f[f'{self.patient_id}/contours/clinical'].keys() if 'ctv' in c and 'fsctv' not in c and 'pctv' not in c]:
                    data = f[f'{self.patient_id}/contours/clinical/{k}'][...]
                    resolution = f[f'{self.patient_id}/contours/clinical/{k}'].attrs['resolution']
                    coordinates = self.original_ct_coordinates if resolution == 'ct' else self.original_dose_coordinates
                    self.contours[k] = Mask(data, resolution, coordinates)

    @property
    def cumulative_dose(self):
        """Compute the cumulative dose volume for all of the beams in the plan.

        Returns
        -------
        ndarray
            3D array containing the cumulative dose volume.
        """
        return np.sum([self.dose[b].data for b in self.dose.keys()], axis = 0)

    @property
    def identify_parallel_capabilities(self):
        
        self.n_threads = self.user_inputs['PARALLELIZATION']['number_of_processors'] 
        if self.n_threads is None: self.n_threads = mp.cpu_count()//2 # assumes that half of the CPUs are available for parallel processing
        if self.n_threads >  mp.cpu_count():
            self.__logger.warning(f"The number of threads ({self.n_threads}) exceeds the number of available CPUs ({mp.cpu_count()}).")
            self.n_threads = mp.cpu_count()//2
            self.__logger.warning(f"Parallel processing will be used with {self.n_threads} threads")
        self.parallelize = True if self.user_inputs['PARALLELIZATION']['parallelize_data_preprocessing'] and self.n_threads > 1 else False   
        
        if not self.parallelize:
            self.n_threads = 1
            self.__logger.info("Preprocessing will be performed sequentially. Consider using parallel processing to speed up the process.")
            return

        # log the available resources
        self.__logger.info(f"Number of CPUs (Virtual): {mp.cpu_count()}")
        
        # get the number of threads ready to perform work concurrently
        if self.user_inputs['PARALLELIZATION']['number_of_processors'] is None and self.parallelize:
            self.n_threads = mp.cpu_count()//2 # assumes that half of the CPUs are available for parallel processing
            self.__logger.info(f"Parallel processing will be used with {self.n_threads} threads")
        else:
            self.n_threads = int(self.user_inputs['PARALLELIZATION']['number_of_processors'])
            if self.parallelize and self.n_threads > mp.cpu_count():
                self.__logger.warning(f"The number of threads ({self.n_threads}) exceeds the number of available CPUs ({mp.cpu_count()}).")
                self.__logger.warning(f"Parallel processing will be used with {mp.cpu_count()} threads")
                self.n_threads = mp.cpu_count()
            elif self.n_threads == 1:
                self.parallelize = False
                self.__logger.info("Parallel processing will not be used since only 1 thread was requested")
        
        if self.n_threads > 1 and self.parallelize:
            self.__logger.info(f"Parallel processing will be used with {self.n_threads} threads")
        else:       
            self.__logger.info("Preprocessing will be performed sequentially. Consider using parallel processing to speed up the process.")
            