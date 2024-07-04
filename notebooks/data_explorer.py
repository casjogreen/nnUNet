import os, json, pickle, logging, traceback
import numpy as np
import pandas as pd
from copy import deepcopy
from dicom_toolbox import DicomToolbox
from utilities import compute_dvh, D, V, create_histogram
from tqdm import tqdm
import multiprocessing as mp

class DataExplorer(DicomToolbox):
    """Class to explore a DICOM-RT dataset. This class helps identify contour name variations, create reports
    of the available contours, and filter patients based on data properties like the number of beams in a plan.

    Created on the Fall of 2022 by Ivan Vazquez in collaboration with Ming Yang. 

    Parameters
    ----------
    DicomToolbox : DicomToolbox
        Super class transfering tools for parsing DICOM-RT datasets

    Copyright 2022 Ivan Vazquez, Fall 2022
    """

    def __init__(self, user_inputs_dir=None, patient_data_directory=None) -> None:
        super().__init__(user_inputs_dir, patient_data_directory)
        self.data_report = None
        self.full_patient_list = None
        self.contour_record_folder = os.path.join('temp','data')
        self.__logger = logging.getLogger(__name__)
        
    def __check_basic_data_report(self, basic_report):
        
        # read all of the folder names in the patient data directory
        folder_names = os.listdir(self.patient_data_directory)
        
        # get the IDs in the basic report
        ids_in_report = list(basic_report.keys())
        
        for i in ids_in_report:
            if i not in folder_names:
                self.__logger.warning(f'Patient ID {i} in the basic report is not in the patient data directory.')
                return False
        
        return True
        
    def create_data_report(self, patient_list=None, save=True, return_val=True, include_extra_information=False):
        """Generate a report of the patient data that includes the number of contours, the prescription 
        dose levels, the number of beam, the beam geometry, and more. Other details like the plan type and 
        diagnosis can be included when the `include_extra_information` flag is set to `True`.

        Parameters
        ----------
        patient_list : list, optional
            List of patient to create a report for, by default None
        save : bool, optional
            Flat to determine if the report should be saved as a CSV file. If `False`, 
            the report is temporarily stored as a class variable, by default True
        return_val : bool, optional
            Activates the returning of the report as a pandas dataframe , by default True
        include_extra_information : bool, optional
            Activates the inclusion of additional details for each patient, by default False

        Returns
        -------
        pandas dataframe
            Report for a given list of patients or all available patients if a list is not given.
        """
      
        # Get a basic report of the patient data
        if self.user_inputs is not None and self.user_inputs['DATA_PREPROCESSING']['generate_new_data_report']:
            basic_report = self.get_dicom_data_report(save =True)
        
        elif os.path.isfile(os.path.join('temp', 'data', 'basic_data_report.pickle')):
            with open(os.path.join('temp', 'data', 'basic_data_report.pickle'), 'rb') as handle:
                basic_report = pickle.load( handle)
                
            if not self.__check_basic_data_report(basic_report): basic_report = self.get_dicom_data_report(save =True)
                
        elif not os.path.isfile(os.path.join('temp', 'data', 'basic_data_report.json')):
            self.__logger.warning('A basic report for the patient data was not found.')
            basic_report = self.get_dicom_data_report(save=True)
             
        # Set flag to False to avoid repeats: This assumes that the code above executed correctly   
        self.user_inputs['DATA_PREPROCESSING']['generate_new_data_report'] = False 
                    
        # Grab IDs for all available patients
        self.full_patient_list = list(basic_report.keys())
        
        # Grab the radiation type
        self.radiation_type = basic_report[self.full_patient_list[0]]['radiation_type'].lower()
        
        # Read the file containing the Rx dose information for the patients
        rx_dose_file_dir = os.path.join("utilities", self.user_inputs['TX_SITE'],  f"rx_plan-type_scales_{self.radiation_type}.csv")
                
        try:
            pat_info = pd.read_csv(rx_dose_file_dir, dtype=str)
        except:
            self.__logger.warning(f'Missing prescription dose file {rx_dose_file_dir}.')
            raise FileNotFoundError(f'Unable to find an specific or infered Rx dose file "{rx_dose_file_dir}".')
                    
        # get the maximum number of beams    
        max_beams = np.max([basic_report[k]["number_of_beams"] for k in basic_report.keys()])
                
        # Dataframe to store the patient information
        df = {"AID":[], "rx_levels":[], "rx_dose":[], "number_of_beams":[]}
        # Add beam information
        df = {**df, **{f"beam_{n+1}_angles":[] for n in range(max_beams)}}
        # Add additional information
        df = {**df, **{"number_of_contours":[], "contours":[], "dx":[],"dy":[],"dz":[],
                       "max_rx_dose":[],  "dose_reference_description": [],
                       "dose_reference_dose":[], "dose_reference_type":[]}}
        # Add isocenter and [V]SAD
        # df = {**df,  **{f"beam_{n+1}_isocenter":[] for n in range(max_beams)}}
        # df = {**df,  **{f"beam_{n+1}_sad":[] for n in range(max_beams)}}
        
        extra = {"plan_type":[],"diagnosis":[],"neck_involved":[], "adaptive":[],}
            
        if patient_list is None: patient_list = self.full_patient_list

        for p in sorted(patient_list):
            
            df["AID"].append(p)
                        
            # Get Rx Dose
            vals = pat_info.loc[pat_info['AID'] == p]["Rx Dose"].values[0]
                    
            df['rx_levels'].append(len([v.strip() for v in vals.split(',')]))
            df["rx_dose"].append(','.join(sorted([v.strip() for v in vals.split(',')])))

            # Beam information: No. beams and angles
            df["number_of_beams"].append(basic_report[p]["number_of_beams"])

            for bn in range(max_beams):
                try:
                    g_ang = basic_report[p]["gantry_angles"][bn]
                    c_ang = basic_report[p]["couch_angles"][bn]
                    df[f"beam_{bn+1}_angles"].append(f"{c_ang}, {g_ang}")
                except:
                    df[f"beam_{bn+1}_angles"].append("none")
                        
            # isocenter and [V]SAD
            # for bn in range(max_beams):
            #     try:
            #         isocenter = basic_report[p]["isocenter"][bn]
            #         if 'vsad' in basic_report[p].keys():
            #             SAD = basic_report[p]["vsad"][bn]
            #         # TODO: Add more info for photon plans
            #         # df[f"beam_{bn+1}_isocenter"].append(",".join([str(x) for x in list(isocenter)]))
            #         # df[f"beam_{bn+1}_sad"].append(",".join([str(x) for x in list(SAD)]))
            #     except:
            #         df[f"beam_{bn+1}_isocenter"].append("none")
            #         df[f"beam_{bn+1}_sad"].append("none")

            # contours
            df["number_of_contours"].append(len(basic_report[p]["contours"].split(',')))
            df["contours"].append(basic_report[p]["contours"])
            
            # record resolution
            df["dx"].append(basic_report[p]["dose_array_resolution"]["dx"])
            df["dy"].append(basic_report[p]["dose_array_resolution"]["dy"])
            df["dz"].append(basic_report[p]["dose_array_resolution"]["dz"])
            
            # add dose reference information
            df["dose_reference_description"].append(basic_report[p]["dose_reference_description"])
            df["dose_reference_dose"].append(basic_report[p]["dose_reference_dose"])
            df["dose_reference_type"].append(basic_report[p]["dose_reference_type"])
            
            # maximum dose
            max_dose = max([float(v.strip()) for v in vals.split(',')]) 
            max_dose = max_dose/100 if max_dose > 1000 else max_dose
            df['max_rx_dose'].append(max_dose)
            
            # Neck involvement, plan type, and diagnosis
            if include_extra_information:
                extra["neck_involved"].append(pat_info.loc[pat_info['AID'] == p]["Neck Involvement"].values[0])
                extra["adaptive"].append(pat_info.loc[pat_info['AID'] == p]["Adaptive"].values[0])
                extra["plan_type"].append(pat_info.loc[pat_info['AID'] == p]["Plan Type"].values[0])
                extra["diagnosis"].append(pat_info.loc[pat_info['AID'] == p]["Diagnosis/Tx Site"].values[0])
                df =  pd.DataFrame({**df, **extra})
        
        self.data_report = pd.DataFrame(df)
        
        if save:
            self.data_report.to_csv(os.path.join('temp','data','data_report.csv'))

        if return_val: return pd.DataFrame(df)

    def __check_conditions(self, c_angle, g_angle, g_condition, c_condition):

        operations = {'(': lambda x,y: x>y, '[': lambda x,y: x>=y,
                      ')': lambda x,y: x<y, ']': lambda x,y: x<=y}

        # check couch angle condition
        if len(c_condition) > 1:
            oper_1, oper_2 = c_condition[0][0],c_condition[1][-1] 
            val_1, val_2 = float(c_condition[0][1:]), float(c_condition[1][:-1])

            cc_met = operations[oper_1](c_angle,val_1) & operations[oper_2](c_angle,val_2)

        else:
            cc_met = float(c_angle) == float(c_condition[0])

        # check gantry angle condition
        if len(g_condition) > 1:
            oper_1, oper_2 = g_condition[0][0],g_condition[1][-1] 
            val_1, val_2 = float(g_condition[0][1:]), float(g_condition[1][:-1])

            gc_met = operations[oper_1](g_angle,val_1) & operations[oper_2](g_angle,val_2)

        else:
            gc_met = float(g_angle) == float(g_condition[0])

        return cc_met & gc_met
    
    def apply_patient_filter(self):
        
        if self.data_report is None: self.create_data_report()
                
        # read the IDs for all of the patients
        pat_list = list(self.data_report['AID'].values)
        
        if not self.user_inputs['DATA_PREPROCESSING']['apply_patient_filter']: return pat_list
        
        # grab the filtering conditions        
        fc = self.user_inputs['DATA_PREPROCESSING']['patient_filter_conditions']
        
        # filter patients based on the number of beams
        number_of_beams = fc['number_of_beams']
        if number_of_beams is not None:
            assert type(number_of_beams) == type([]) or type(number_of_beams) == type(int(1)), "The number of beams must be a list or an integer."
        
            # if the number of beams is an integer, convert it to a list
            if type(number_of_beams) == type(int(1)): number_of_beams = [number_of_beams]           
            assert  all([type(n) == type(int(1)) for n in number_of_beams]), "The elements in the number of beams list must be integers."
           
            pat_list = [p for p in pat_list if self.data_report.loc[self.data_report['AID'] == p]['number_of_beams'].values[0] in number_of_beams]
                  
        # check if the number of prescription levels is a list or an integer
        number_of_rx_lvls = fc['prescription_levels']
        if number_of_rx_lvls is not None:
            assert type(number_of_rx_lvls) == type([]) or type(number_of_rx_lvls) == type(int(1)), "The number of prescription levels must be a list or an integer."
        
            # if the number of prescription levels is an integer, convert it to a list
            if number_of_rx_lvls is not None and type(number_of_rx_lvls) == type(int(1)): number_of_rx_lvls = [number_of_rx_lvls]
            
            pat_list = [p for p in pat_list if self.data_report.loc[self.data_report['AID'] == p]['rx_levels'].values[0] in number_of_rx_lvls]
        
        # apply filter based on the optimization type
        if fc['optimization_type'] is not None:
            rx_to_aid_matching = pd.read_csv(os.path.join("utilities",self.user_inputs['TX_SITE'], f"rx_plan-type_scales_{self.radiation_type}.csv")).set_index('AID')
            plan_type_info = rx_to_aid_matching.to_dict()['Plan Type']
            pat_list = [p for p in pat_list if fc['optimization_type'].lower() == plan_type_info[int(p)].lower()]
                
        # apply filtering based on gantry and couch angle conditions
        if  fc['couch_angle_ranges'] is None or fc['gantry_angle_ranges'] is None: return pat_list
        
        unique_quadrants = fc['all_unique_quadrants']
        gantry_angle_conditions = [gc.split(',') for gc in fc['gantry_angle_ranges']]
        couch_angle_conditions = [cc.split(',') for cc in fc['couch_angle_ranges']]
        nb = fc['number_of_beams']
    
        assert len(gantry_angle_conditions) == len(couch_angle_conditions), "The number of gantry and couch angle conditions must be the same." 
        if unique_quadrants is None: unique_quadrants = False
        
        selected_patients = []

        for p in pat_list:
            
            p_info = self.data_report.loc[self.data_report['AID'] == p]
            conditions_met = 0
            
            if not unique_quadrants:

                for b in range(nb):
                    angles = [float(a) for a in p_info[f"beam_{b+1}"].values[0].split(',')]
                
                    c_angle, g_angle = angles
                    for gc, cc in zip(gantry_angle_conditions, couch_angle_conditions):
                        if self.__check_conditions(c_angle, g_angle, gc, cc): 
                            conditions_met+=1
                            break
    
            elif unique_quadrants:

                for gc, cc in zip(gantry_angle_conditions, couch_angle_conditions):
                    for b in range(nb):
                        angles = [float(a) for a in p_info[f"beam_{b+1}_angles"].values[0].split(',')]
                        c_angle, g_angle = angles         
                        if self.__check_conditions(c_angle, g_angle, gc, cc): 
                            conditions_met+=1
                            break
                                
            if conditions_met == nb: selected_patients.append(p)

        return selected_patients   
    
    def __get_all_available_target_names(self, pat_id):
        
        # grab the type of target
        target_type = self.user_inputs['TYPE_OF_TARGET_VOLUME'].lower()
        
        # grab prescription levels for patient
        p_info = self.data_report.loc[self.data_report['AID'] == pat_id]
        
        # Grab available contours for current patients
        available_contours = p_info["contours"].values[0].split(',')
        
        # Grab all contour names with targets in it
        ignore_values = ['fs', 'pctv', 'v40br', 'exp', 'isoline', 'sub', 'low', 'push','ring','zctv', 'zptv', 'mm', "_ap", "boost", 'dnu']  #TODO: add more values to ignore
    
        return [ac for ac in available_contours if target_type in ac and all([x not in ac for x in ignore_values])]
                   
    def find_targets(self, pat_id):
        """Function to find the target volumes for a given patient. 
        
        Assumptions:
        1. The target name normally contains the prescription dose level. 
        2. The target name normally contains the type of target volume (STV, CTV, PTV, GTV)
        3. Prostate cases have one target volume
        
        """
        
        # grab the type of target
        target_type = self.user_inputs['TYPE_OF_TARGET_VOLUME'].lower()
                            
        # sanity checks
        self.create_data_report() 
        if type(pat_id) != type(''): pat_id = str(pat_id)

        # grab prescription levels for patient
        p_info = self.data_report.loc[self.data_report['AID'] == pat_id]
                
        # Grab Rx dose values
        rx_vals = sorted([rx.strip() for rx in p_info['rx_dose'].values[0].split(',')], reverse=True) 
        
        # Get all available target names
        available_target_names = self.__get_all_available_target_names(pat_id)  
                
        # Grab cummulative dose
        self.parse_dicom_files(patient_id = pat_id, mask_names_only=True)
        cum_dose = np.sum([self.dose[k].data for k in self.dose.keys()], axis=0)
           
        # Load the variations for contour names
        with open(os.path.join("utilities",self.user_inputs['TX_SITE'],"structure_name_variations.json"),"r") as f:
            contour_variations = json.load(f)

        good_names = contour_variations[target_type][:]
        with open(os.path.join("utilities",self.user_inputs['TX_SITE'], f"junk_symbols.json"),"r") as f:
            junk_symbols = json.load(f)['target']
            
        # crude way of generating accepatable names for targets
        links, suffix = ["_","-",""," "], ['cge','gy',"[cgy]", "cgy",""]
        
        if self.user_inputs['TYPE_OF_TARGET_VOLUME'].lower() == 'stv':
            target_name = p_info['dose_reference_description'].values[0].lower()
            links+=[' custom', '_custom', 'custom_' ] # very specific junk symbols for a patient...sorry
                    
        for rx in [r for r in rx_vals]:
            for v in [f"{vv}{l}{n}" for vv in contour_variations[target_type] for l in links for n in [i for i in range(1,len(rx_vals)+1)]+[""]]:
                for s in suffix:
                    good_names += [f"{v}{l1}{r}{l2}{s}" for l1 in links for l2 in links for r in [f'{rx}', f'({rx})', ""]]
                    good_names += [f"{target_type}{rx}{l}{n}" for l in links for n in range(1,len(rx_vals)+1)]
                            
        # Find candidates for the targets
        found, candidates = 0, []
        
        # Search for matches without removing junk symbols
        for n in available_target_names:
            for gn in good_names:
                if gn.strip() == n:
                    candidates.append(n)
                    found+=1
                    break
        
        # log the name of all of the targets found
        self.__logger.info(f"{target_type} names found for pat-{pat_id}: {available_target_names}")
                
        # remove junk symbols and search again      
        for n in available_target_names:
            n_new = deepcopy(n)
            for i in junk_symbols:
                
                n_new_sr = n.replace(i,"").strip() # single replacement
                n_new = n_new.replace(i,"").strip() # cummulative replacement
                
                for gn in good_names:
                    if gn.strip() == n_new_sr.strip():
                        if n not in candidates: candidates.append(n)
                        found+=1
                        break
                    if gn.strip() == n_new.strip():
                        if n not in candidates: candidates.append(n)
                        found+=1
                        break  
                                    
        # check one of the candidates matches the target name - Assumes that prostate cases have one target
        if self.user_inputs['TX_SITE'].lower() == 'prostate':
            for c in candidates:
                if c == target_name:
                    target_info = {c:{'rx_value': np.round(float(rx_vals[0]),4), 'D98':None}}
                    return target_info, available_target_names
                                        
        if len(rx_vals) > found: 
            self.__logger.warning(f"Failed to find matching set of {target_type} contours. Check pat-{pat_id}.")
            self.__logger.info(f"Expected {len(rx_vals)} and found {found} for pat-{pat_id}")
            
        # check if there is only one prescription level
        if len(rx_vals) == 1: 
            rx =  np.round(float(rx_vals[0]),4) if float(rx_vals[0]) < 1000 else np.round(float(rx_vals[0])/100,4)
            if target_type in candidates:
                return {target_type:{'rx_value': rx, 'D98':None}}, available_target_names 
            elif len(candidates) == 1:
                target_info ={candidates[0]:{'rx_value': rx, 'D98':None}}
                return target_info, available_target_names
            elif len(candidates) > 1:
                for c in candidates:
                    found_targets = [c for c in candidates if str(rx_vals[0]) in c]
                    if found_targets != []: 
                        target_info = {found_targets[0]:{'rx_value': rx, 'D98':None}}
                        return target_info, available_target_names
                
        # Final match for targets based on dose coverage
        target_info = {}
        
        if self.user_inputs['TX_SITE'].lower() == 'breast':
            use_status = {rx:False for rx in rx_vals}
            for rx in rx_vals:
                for c in candidates:
                    if str(rx) in c:
                        if float(rx) > 1000: 
                            target_info[c] = {'rx_value': np.round(float(rx)/100,4), 'D98':None}
                        else:
                            target_info[c] = {'rx_value': np.round(float(rx),4), 'D98':None}
                            
                        use_status[rx] = True
            # check if all prescription levels were used
            if all([use_status[rx] for rx in use_status.keys()]): return target_info, available_target_names
        else:
            for rx in rx_vals:
                for c in candidates:
                    if str(rx) in c:
                        if float(rx) > 1000: 
                            target_info[c] = {'rx_value': np.round(float(rx)/100,4), 'D98':None}
                        else:
                            target_info[c] = {'rx_value': np.round(float(rx),4), 'D98':None}
                        break
                         
        if len(rx_vals) == len(target_info.keys()): 
            return target_info, available_target_names
        
        # Transform cGy to Gy
        rx_vals = sorted([float(rx) if float(rx) < 1000 else float(rx)/100 for rx in rx_vals], reverse=True)
        
        # Remove values already used
        rx_used = [target_info[c]['rx_value'] for c in target_info.keys()]
        for rx in rx_used: rx_vals.remove(rx)
                
        candidates = [c for c in candidates if c not in list(target_info.keys())]
        
        if len(candidates) == 0: 
            self.__logger.error(f"""Failed to find a matching set of {target_type} contours for pat-{pat_id}.
                                    Expected {len(rx_vals)} and found {len(target_info.keys())}.
                                    The following contours were found: {available_target_names}
                                    The available target names are: {available_target_names}
                                    The remaining prescription levels are: {rx_vals}""")     
            raise AssertionError(f"Failed to find a matching set of {target_type} contours for pat-{pat_id}.")
                        
        # Start comparison between D98 value and Rx for each candidate   
        D98 = {} # initialize dictionary for D98 values    
        if self.echo_level > 1: self.__logger.info(f"""Starting comparison between D98 and Rx for pat-{pat_id}.
                                                       The candidate target(s) name(s) are: {', '.join(candidates)}.""") 
                
        for c in candidates:
            
            # compute the DVH for the target volume
            target = self.parse_structure_files(patient_id = pat_id, mask_names = c, resolution = 'dose')[c].data
            D98_val = np.percentile(cum_dose[np.where(np.round(target) > 0)], 2)            
            D98[D98_val] = c
            
        # sort the D98 values from highest to lowest        
        vals = sorted(list(D98.keys()), reverse=True)
                                        
        if len(rx_vals) == 1: # if there is only one prescription level
            target_info[D98[np.max(vals)]] = {'rx_value': np.round(rx_vals[0],4), 'D98':np.round(np.max(vals),4)}
            
        elif len(vals) == len(rx_vals): # if there is a one-to-one match between prescription levels and D98 values
            for n,v in enumerate(vals): 
                target_info[D98[v]] = {'rx_value': np.round(rx_vals[n],4), 'D98':np.round(v,4)}
        else:
            for rx in rx_vals:                
                diff = [np.abs(rx-val) for val in vals]
                target_info[candidates[np.argmin(diff)]]= {'rx_value': np.round(rx,4), 'D98':np.round(vals[np.argmin(diff)],4)}

        return target_info, available_target_names
    
    def __get_available_oars(self, pat_id):
                
        # grab prescription levels for patient
        p_info = self.data_report.loc[self.data_report['AID'] == pat_id]
        
        # Grab available contours for current patients
        return sorted([n.strip() for n in p_info["contours"].values[0].split(',') if not any([x in n for x in ['ctv', 'ptv', 'gtv','dose','artifact']])])

    def find_oars(self, pat_id, contours = [], desperate_mode = True):
           
        # sanity checks
        self.create_data_report() 
        if type(pat_id) != type(''): pat_id = str(pat_id)
           
        # prepare dictionary with desired contours
        if contours == []: 
            contours = {c:0 for c in sorted(list(self.user_inputs['DATA_PREPROCESSING']["oars"]["name_and_voxel_value_pairs"].keys()))}
        elif type(contours) == type([]):
            contours = {c:0 for c in contours}
                
        # Load the variations for contour names
        with open(os.path.join("utilities",self.user_inputs['TX_SITE'], "structure_name_variations.json"),"r") as f:
            contour_variations = json.load(f)

        # Grab available contours for current patients
        available_contours = self.__get_available_oars(pat_id)
        
        # items added to the name of contours that make it difficult to identify them
        with open(os.path.join("utilities",self.user_inputs['TX_SITE'], "junk_symbols.json"),"r") as f:
            junk_symbols = json.load(f)['oar']
        lr_variations = {'left': ['l', 'lt', 'left', 'lft', 'lf', 'lef', 'le', 'i'], # variations of left
                         'right':['r', 'rt', 'right','rgt','righ', 'rig', 'rgh']} # variations of right
        links = ['_','-'," ",""] # types of links used for word elements in contour names
                
        for c in contours:  

            side =  lr_variations[c.split('_')[0]] if 'left' in c or 'right' in c else [""] # symbols to attach to organs with left or right in the name
            gland = [""] if 'gland' not in c else ['gl', 'gland', 'g', "glnd", ""] # symbols to attach to organs with gland in the name
            links2 = links if 'gland' in c else [""] # links to attach gland to name 
            k = '_'.join([kk for kk in c.split('_') if kk not in ['left', 'right', 'gland']]) # root name of the contour (removing left, right, and gland)
              
            # generate name variations
            name_variations = [f"{n}{l1}{g}{l2}{s}" for n in contour_variations[k] for l1 in links for l2 in links2 for s in side for g in gland]
            name_variations += [f"{g}{l1}{n}{l2}{s}" for n in contour_variations[k] for l1 in links for l2 in links2 for s in side for g in gland]
            name_variations += [f"{s}{l1}{n}{l2}{g}" for n in contour_variations[k] for l1 in links for l2 in links2 for s in side for g in gland]
            name_variations += [f"{s}{l1}{g}{l2}{n}" for n in contour_variations[k] for l1 in links for l2 in links2 for s in side for g in gland]
                            
            for v in name_variations:

                if contours[c] != 0: break

                for n in available_contours:
                    if n == v: contours[c] = n.strip()

                for n in available_contours:
                    if contours[c] == 0:
                        n_new = "".join([i for i in n if not i.isdigit()])
                        if n_new.strip() == v: contours[c] = n.strip()

                for n in available_contours:
                    if contours[c] == 0:
                        n_new = "".join([i for i in n.strip() if i not in ['_','/','.','-']])
                        n_new = "".join([i for i in n_new if not i.isdigit()])
                        if n_new.strip() == v: contours[c]= n.strip()
                        
                for n in available_contours:
                    if desperate_mode and contours[c] == 0:
                        for item in junk_symbols:
                            n_new = n.strip().replace(item,"")
                            n_new = "".join([i for i in n_new.strip() if i not in ['_','/','.','-']])
                            n_new = "".join([i for i in n_new if not i.isdigit()])
                            if n_new.strip() == v: contours[c] = n.strip()    
                               
        return contours, available_contours
    
    def find_targets_and_oars(self, patient_list, name=''):
        
        selected_oars_report = {p:'NA' for p in patient_list} 
        selected_targets_report = {p:'NA' for p in patient_list} 
        full_oars_report = {p:'NA' for p in patient_list} 
        full_targets_report = {p:'NA' for p in patient_list} 
        
        if self.echo_progress and not self.user_inputs['PARALLELIZATION']['parallelize_data_preprocessing']: 
            p_list = tqdm(patient_list) 
        else:
            p_list = patient_list
            self.__logger.info(f'Patient list assigned to {mp.current_process().name}: {",".join(patient_list)}')
        
        for n, p in enumerate(p_list):
            try:
                target_info = self.find_targets(p)
            except:
                target_info = 'NA', self.__get_all_available_target_names(p)
                self.__logger.warning(f'Failed to find target volumes for pat-{p}')
            
            try:
                oar_info = self.find_oars(p)
            except:
                oar_info = 'NA', self.__get_available_oars(p)
                self.__logger.warning(f'Failed to find OARs for patient {p}')
            
            selected_targets_report[p], full_targets_report[p] = target_info
            selected_oars_report[p], full_oars_report[p] = oar_info
            self.__logger.info(f'Finished finding OARs and targets for pat-{p}')
            if n < len(patient_list)-1: self.__logger.info(f'Starting with {patient_list[n+1]}')
            
        with open(os.path.join(self.contour_record_folder,f"selected_oars_report{name}.json"), "w") as f:
            f.write(json.dumps(selected_oars_report, indent=4))
        
        with open(os.path.join(self.contour_record_folder,f"full_oars_report{name}.json"), "w") as f:
            f.write(json.dumps(full_oars_report, indent=4))
        
        with open(os.path.join(self.contour_record_folder,f"selected_targets_report{name}.json"), "w") as f:
            f.write(json.dumps(selected_targets_report, indent=4))

        with open(os.path.join(self.contour_record_folder,f"full_targets_report{name}.json"), "w") as f:
            f.write(json.dumps(full_targets_report, indent=4))
        
    def generate_contour_report(self, patient_list = None, output_folder = None):
       
        # create data report
        self.create_data_report() 

        if patient_list is None: patient_list = self.full_patient_list

        if type(patient_list) != type([]): patient_list = [patient_list]
        
        # check if the output directory exists
        if output_folder is not None: self.contour_record_folder = output_folder
        if not os.path.exists(self.contour_record_folder): os.makedirs(self.contour_record_folder)
        
        # use multiprocessing
        if self.user_inputs['PARALLELIZATION']['parallelize_data_preprocessing']:
            
            # check parallelization capabilities
            if self.n_threads is None: self.identify_parallel_capabilities

            # split patient list using np.array_split
            chunked_patients = np.array_split(patient_list, self.n_threads) 
            chunked_patients = [x for x in chunked_patients if x.size != 0]
            
            processes = [] # initialize a list to store the processes
                            
            # create a process for each thread
            for n, patients in enumerate(chunked_patients):
                p = mp.Process(target=self.find_targets_and_oars, args=(patients, n,), name=f"Process-{n}")
                p.start()
                processes.append(p)
            
            for p in processes:
                p.join()   
                
            # merge data reports and remove temp files
            selected_targets_report = {}
            selected_oars_report = {}
            full_targets_report = {}
            full_oars_report = {}
            
            for n in range(len(chunked_patients)):
                with open(os.path.join(self.contour_record_folder,f"selected_oars_report{n}.json"), "r") as f:
                    selected_oars_report.update(json.load(f))
                with open(os.path.join(self.contour_record_folder,f"selected_targets_report{n}.json"), "r") as f:
                    selected_targets_report.update(json.load(f))
                with open(os.path.join(self.contour_record_folder,f"full_oars_report{n}.json"), "r") as f:
                    full_oars_report.update(json.load(f))
                with open(os.path.join(self.contour_record_folder,f"full_targets_report{n}.json"), "r") as f:
                    full_targets_report.update(json.load(f))
                os.remove(os.path.join(self.contour_record_folder,f"selected_oars_report{n}.json"))
                os.remove(os.path.join(self.contour_record_folder,f"selected_targets_report{n}.json"))
                os.remove(os.path.join(self.contour_record_folder,f"full_oars_report{n}.json"))
                os.remove(os.path.join(self.contour_record_folder,f"full_targets_report{n}.json"))
                
            with open(os.path.join(self.contour_record_folder,f"selected_oars_report.json"), "w") as f:
                f.write(json.dumps(selected_oars_report, indent=4))
                
            with open(os.path.join(self.contour_record_folder,f"selected_targets_report.json"), "w") as f:
                f.write(json.dumps(selected_targets_report, indent=4))
                
            with open(os.path.join(self.contour_record_folder,f"full_oars_report.json"), "w") as f:
                f.write(json.dumps(full_oars_report, indent=4))
            
            with open(os.path.join(self.contour_record_folder,f"full_targets_report.json"), "w") as f:
                f.write(json.dumps(full_targets_report, indent=4))
                
        else:
            self.find_targets_and_oars(patient_list)
            
    def calculate_dose_properties(self, pat_id_list, target_metrics, oar_metrics, store_dvh_data, name=''):
                    
        # read the target and oar information for the patient data
        with open(os.path.join('utilities', self.user_inputs['TX_SITE'], 'selected_targets_report.json'), 'r') as f:
            target_info_full = json.load(f)
            
        with open(os.path.join('utilities', self.user_inputs['TX_SITE'], 'selected_oars_report.json'), 'r') as f:
            oar_info_full = json.load(f)
            
        # read the information for the patient data
        if self.user_inputs['DIRECTORIES']['patient_info'] is not None:
            patient_data = pd.read_csv(self.user_inputs['DIRECTORIES']['patient_info']) 
        else:
            patient_data = pd.read_csv(os.path.join('utilities', self.user_inputs['TX_SITE'], f'rx_plan-type_scales_{self.radiation_type}.csv'))
            
        # convert the AID colume into strings
        patient_data['AID'] = patient_data['AID'].astype(str)
            
        target_analysis = {}
        oar_analysis = {}
        apply_scaling = self.user_inputs['DATA_PREPROCESSING']['dose']['apply_dose_scale']  
        
        # use tqdm to track progress
        iterator = tqdm(enumerate(pat_id_list, start=1), total=len(pat_id_list), desc='Calculating dose properties') if not self.parallelize else enumerate(pat_id_list, start=1)

        for n, p in iterator:
                        
            try:
            
                # grab scales for dose volumes
                dose_scale = patient_data.loc[patient_data['AID'] == p, 'dose_scale'].iloc[0] if apply_scaling else 1
                                
                # get cumulative dose
                self.parse_dicom_files(p, mask_resolution='dose', mask_names_only=True)
                cum_dose = self.cumulative_dose * dose_scale
                
                # get the voxel volume in cm^3
                bn = list(self.dose.keys())[0]
                vox_vol = self.dose[bn].coordinates.dx * self.dose[bn].coordinates.dy * self.dose[bn].coordinates.dz/ 1000 # cm^3
                rel_volume_at_max_dose = self.user_inputs['EVALUATION_PARAMETERS']['vol_for_max_dose_in_cc']/vox_vol
                
                # grab the target and oar info for the patient
                target_info = target_info_full[str(p)]            
                oar_info = oar_info_full[str(p)]
                
                # grab the target names and prescription levels
                rx_vals = sorted([float(v['rx_value']) for k,v in target_info.items()], reverse=True)
                max_rx_dose = np.max(rx_vals)
                targets = [k for n,k in enumerate(target_info.keys()) if float(target_info[k]['rx_value']) == rx_vals[n]]
                
                if store_dvh_data and not os.path.exists(os.path.join('temp','data','dvh')): os.makedirs(os.path.join('temp','data','dvh'))
                
                new_target_info = {}
                new_oar_info = {}
                dvh_data = {}
                            
                if target_metrics is not None:
                    
                    for t in targets:
                                        
                        # add target name and Rx dose level
                        new_target_info[t] = {'rx_value': target_info[t]['rx_value']}
                        
                        target = self.__read_contour(p, t)
                            
                        # compute the cummulative dose
                        d,v = compute_dvh(target, cum_dose, maxDose=85, binSize=0.1)
                        dvh_data['dose'] = d
                        dvh_data[t] = v
                
                        # compute dose metrics
                        new_target_info[t].update(self.__compute_dose_metrics(d, v, cum_dose, target, target_info[t]['rx_value'], rel_volume_at_max_dose, target_metrics))
                    
                # check if OAR metrics are empty  
                if oar_metrics is not None: 
                                                            
                    for oar in oar_info.keys():

                        if oar_info[oar] != 0:
                            
                            oar_mask = self.__read_contour(p, oar_info[oar])
                            
                            d,v = compute_dvh(oar_mask, cum_dose, maxDose=85, binSize=0.1)
                            
                            new_oar_info[oar] = self.__compute_dose_metrics(d, v, cum_dose, oar_mask, max_rx_dose, rel_volume_at_max_dose, oar_metrics)
                                        
                            dvh_data[oar] = v
                        else:
                            dvh_data[oar] = [np.nan]*len(dvh_data['dose'])
                    
                if not self.parallelize: self.__logger.info(f'Finished calculating dose properties for patient {p}: {n} our of {len(pat_id_list)}')  
                target_analysis[p] = new_target_info
                oar_analysis[p] = new_oar_info
                
                # save dvh data as csv
                if store_dvh_data:
                    df = pd.DataFrame(dvh_data).to_csv(os.path.join('temp','data', 'dvh', f'dvh_data_p-{p}.csv'))
                        
            except Exception as e:
                self.__logger.error(f'Failed to calculate dose properties for patient {p}')
                self.__logger.error(traceback.format_exc())
            
            # # save dvh data as csv
            # if store_dvh_data:
            #     df = pd.DataFrame(dvh_data).to_csv(os.path.join('temp','data', 'dvh', f'dvh_data_p-{p}.csv'))
                
        # save the results
        with open(os.path.join('temp','data',f'target_dose_analysis{name}.json'), 'w') as f:
            json.dump(target_analysis, f, indent=4)
            
        with open(os.path.join('temp','data',f'oar_dose_analysis{name}.json'), 'w') as f:
            json.dump(oar_analysis, f, indent=4)
     
    def get_dose_property_reports(self, target_metrics = {'D':[95,98,99, 'max', 'mean'],'V':[95, 100, 105]}, 
                                  oar_metrics = {'D':[1, 2, 'mean', 'max'], 'V':[10, 20, 30, 40, 50, 60, 70, 80, 90]}, 
                                  mrn_info_dir = None, save_dvh_data = True):
        
        # de = DataExplorer(user_inputs_dir)
        self.__logger.info('Volume at maximum dose specified in "EVALUATION_PARAMETERS": {} cm^3'.format(self.user_inputs['EVALUATION_PARAMETERS']['vol_for_max_dose_in_cc']))
        
        # create data report
        self.create_data_report() 

        if self.n_threads is None: self.identify_parallel_capabilities

        # use multiprocessing
        if self.parallelize:
            
            # check parallelization capabilities
            if self.n_threads is None: self.identify_parallel_capabilities()

            # parallelize execution of function
            chunked_patients = np.array_split(self.full_patient_list, self.n_threads) 
            chunked_patients = [x for x in chunked_patients if x.size != 0]

            processes = [] # initialize a list to store the processes
            
            # create a process for each thread
            for process_name, patients in enumerate(chunked_patients):
                p = mp.Process(target=self.calculate_dose_properties, args=(patients, target_metrics, oar_metrics, save_dvh_data, str(process_name), ))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            # combine all of the results
            target_analysis = {}
            for n in range(len(chunked_patients)):
                with open(os.path.join('temp','data',f'target_dose_analysis{n}.json'), 'r') as f:
                    target_analysis.update(json.load(f))        
                os.remove(os.path.join('temp','data',f'target_dose_analysis{n}.json'))
                    
            with open(os.path.join('temp','data','target_dose_analysis.json'), 'w') as f:
                json.dump(target_analysis, f, indent=4)
                
            oar_analysis = {}
            for n in range(len(chunked_patients)):
                with open(os.path.join('temp','data',f'oar_dose_analysis{n}.json'), 'r') as f:
                    oar_analysis.update(json.load(f))        
                os.remove(os.path.join('temp','data',f'oar_dose_analysis{n}.json'))
                
            with open(os.path.join('temp','data','oar_dose_analysis.json'), 'w') as f:
                json.dump(oar_analysis, f, indent=4)
            
        else:
            self.calculate_dose_properties(self.full_patient_list, target_metrics, oar_metrics, save_dvh_data)  
            
    def __read_contour(self, patient_id, contour):
        if self.user_inputs is None or self.user_inputs['DIRECTORIES']['raw_patient_data'].split('.')[-1] not in ['h5', 'hdf5']:
            data = self.parse_structure_files(mask_names=contour, resolution='dose')[contour].data
        elif self.user_inputs['DIRECTORIES']['raw_patient_data'].split('.')[-1] in ['h5', 'hdf5']:
            self.read_data_from_hdf5(patient_id=patient_id, contours_type=self.user_inputs['DATA_PREPROCESSING']['type_of_contours'].lower())
            data = self.contours[contour].data[:]
        return data
    
    def __compute_dose_metrics(self, d, v, dose, contour, rx_dose, rel_volume_at_max_dose, desired_metrics):
        
        results = {}
        
        if desired_metrics is None: return results
        
        for metric in desired_metrics.keys():
            for val in desired_metrics[metric]:
                if val == 'mean':
                    results[f'{metric}_mean'] = np.sum(np.multiply(contour, dose))/np.count_nonzero(contour)
                elif val == 'max':
                    Vmax = 100*(rel_volume_at_max_dose/np.count_nonzero(contour))
                    results[f'{metric}_{val}'] = D(Vmax, d,v)
                elif metric == 'D':
                    results[f'{metric}_{val}'] = D(val, d,v)
                elif metric == 'V':
                    results[f'{metric}_{val}'] = V(rx_dose*val/100, d,v)
                    
        return results
    
    def get_dose_metric_report_as_csv(self, target_analysis_dir, metric, mrn_file_dir = None, max_rx_levels = None):
        
        with open(target_analysis_dir, 'r') as f:
            target_analysis = json.load(f)
            
        if max_rx_levels is None: max_rx_levels = np.max([len(list(target_analysis[p].keys())) for p in target_analysis.keys()])
                
        for index in range(max_rx_levels):
                    
            target_info = {'AID':[], 'rx_value':[], 'target_name':[], metric:[]}
                    
            if mrn_file_dir is not None:
                mrn_info = pd.read_csv(mrn_file_dir)
                target_info['TID'] = []

            for p in target_analysis.keys():
                
                try:
                    target_info['rx_value'].append(target_analysis[p][list(target_analysis[p].keys())[index]]['rx_value'])
                    target_info['AID'].append(p)
                    target_info['target_name'].append(list(target_analysis[p].keys())[index])
                    target_info[metric].append(target_analysis[p][list(target_analysis[p].keys())[index]][metric])
                    if mrn_file_dir is not None:
                        target_info['TID'].append(mrn_info.loc[mrn_info['AID'] == p, 'TID'].iloc[0])
                except:
                    self.__logger.info(f'Patient {p} does not have a target with index {index}')

            target_info = pd.DataFrame(target_info)
                    
            # save the results as csv
            target_info.to_csv(os.path.join('temp','data',f'target-{index+1}_{metric}_info.csv'), index=False)

            create_histogram(target_info[metric], f'histogram_of_{metric}_target-{index+1}', xlable=metric.capitalize(), fig_dir= None)