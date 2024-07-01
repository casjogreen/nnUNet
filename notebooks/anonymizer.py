import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from tkinter.font import Font
import dicognito.anonymizer, os, shutil, datetime, numpy as np, pandas as pd
import datetime
from pydicom import dcmread
import sys

class DicomAnonymizer():
    
    tags = {'PatientID': 0, # get from table eventually,
            'InstitutionName': 'ANONYMIZED',
            'InstitutionAddress': 'ANONYMIZED',
            'Manufacturer': 'ANONYMIZED',
            'ManufacturerModelName': 'ANONYMIZED',
            'BodyPartExamined': 'ANONYMIZED',
            'StudyDescription': 'DL Dose Uncertainty',
            'RTPlanDescription': 'ORIGINAL',
            'StationName': 'ANONYMIZED',
            'AccessionNumber' : 'ANONYMIZED',
            'StudyID': 0}
    
    def __init__(self, datDir, patListDir, outDir = None, study_id = 'BL-Anonymizer-Tool'):
        self.startDate = datetime.date(1910, 1, 1)
        self.endDate = datetime.date(2020, 1, 1)
        self.datDir = datDir
        self.outDir = outDir
        self.patListDir = patListDir
        self.fNames = []
        self.uids = {}
        self.study_id = study_id
        
    def parse_for_organization(self):
        """Parse the data directory to categorize files by patient ID and modality."""
        self.patient_folders = {}

        for root, _, files in os.walk(self.datDir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    ds = dcmread(file_path)
                    modality = ds.Modality.lower()
                    patient_id = ds.PatientID
                    if patient_id not in self.patient_folders:
                        self.patient_folders[patient_id] = {}
                    if modality not in self.patient_folders[patient_id]:
                        self.patient_folders[patient_id][modality] = []
                    self.patient_folders[patient_id][modality].append(file_path)
                except:
                    continue
    
    def parse_data_folder(self):       
        allFiles = os.listdir(self.datDir)
        patInfo = pd.read_csv(self.patListDir)
        self.patTID = patInfo['TID'].values
        self.patAID = patInfo['AID'].values
        
        # Find data folders for each patient in master list
        self.datFldrList = {}
        for n, TID in enumerate(self.patTID):
            fldr = os.path.join(self.datDir, str(TID))
            if os.path.isdir(fldr): self.datFldrList[TID]={'dir':fldr,'AID':self.patAID[n],'file_types':[]}
        
        # Create a directory for each listed and found patient
        for k in self.datFldrList.keys():
            outDir = self.outDir or os.path.join(self.datDir, 'anonymized')
            path = os.path.join(outDir, str(self.datFldrList[k]['AID']))
            if os.path.exists(path):
                shutil.rmtree(path)
            os.mkdir(path)
            
            for mod in ['ct', 'rtstruct', 'rtplan', 'rtdose']:
                os.mkdir(os.path.join(path, mod))
        
        # Explore folder content
        for k in self.datFldrList.keys():
            self.datFldrList[k]['content_dirs'] =[]
            # walk through all files in the folder and append DICOM files to list
            for root, dirs, files in os.walk(self.datFldrList[k]['dir']):
                for file in files:
                    try:
                        ds = dcmread(os.path.join(root, file))
                        if ds.data_element('Modality').value in ['CT', 'RTSTRUCT', 'RTPLAN', 'RTDOSE']:
                            self.datFldrList[k]['content_dirs'].append(os.path.join(root, file))
                    except:
                        pass

            for f in self.datFldrList[k]['content_dirs']:
                try:
                    ds = dcmread(os.path.join(self.datFldrList[k]['dir'], f))
                    modality = ds.data_element('Modality').value.lower()
                    self.datFldrList[k]['file_types'].append(modality)
                except:
                    pass
                            
    def organize_dicom_files(self):
        self.parse_for_organization()  # Parse data for organization without dependencies

        organized_dir = os.path.join(self.datDir, 'Organized')
        os.makedirs(organized_dir, exist_ok=True)

        for patient_id, patient_data in self.patient_folders.items():
            patient_dir = os.path.join(organized_dir, patient_id)
            os.makedirs(patient_dir, exist_ok=True)  # Ensure patient directory exists

            for modality, files in patient_data.items():
                mod_dir = os.path.join(patient_dir, modality.lower())
                os.makedirs(mod_dir, exist_ok=True)  # Ensure modality directory exists
                for file in files:
                    shutil.copy2(file, mod_dir)
    
    def prepare_output_file_name(self, dataSet, info):
        '''
        Generate an output file directory describing the patient ID for 
        all files and beam number of slice number for dose and CT files, 
        respectively.
        '''
         
        ds = dataSet
        
        modality = ds.data_element('Modality').value
        AID = info['AID']
        
        if modality == 'CT': 
            sliceNum = ds.data_element('InstanceNumber').value
            outName = f'CT_p{AID}_s{sliceNum}.dcm' 
        elif modality == 'RTDOSE':
            try:
                bn = int(ds.ReferencedRTPlanSequence[0][('300c','0020')][0][('300c','0004')][0][('300c', '0006')].value)
                outName = f'RTD_p{AID}_b{bn}.dcm'   
            except: 
                if ds.DoseSummationType.lower() == 'plan': 
                    outName = f'RTD_p{AID}_plan-dose.dcm'
        elif modality == 'RTPLAN':
            if info['file_types'].count('rtplan') > 1:
                outName = f'RTP_p{AID}_{self.rt_plan_file_number}.dcm'
                self.rt_plan_file_number += 1
            else:
                outName = f'RTP_p{AID}.dcm'

        elif modality == 'RTSTRUCT':
            if info['file_types'].count('rtstruct') > 1:
                outName = f'RTS_p{AID}_{self.rt_struct_file_number}.dcm'
                self.rt_struct_file_number += 1
            else:
                outName = f'RTS_p{AID}.dcm'
            
        return os.path.join(str(AID),modality.lower(), outName)
     
    def anonymize_dicom_files(self):
        
        outputDir = self.outDir or os.path.join(self.datDir, 'anonymized')
        if not os.path.isdir(outputDir): os.mkdir(outputDir)
        
        self.parse_data_folder()
        
        for k in self.datFldrList.keys():
            
            # 0.1 initialize anonymizer
            anonymizer = dicognito.anonymizer.Anonymizer()

            # 0.2 initialize structure file number
            self.rt_struct_file_number = 1
            self.rt_plan_file_number = 1
            
            for f in self.datFldrList[k]['content_dirs']:
                # 1. Read DICOM file
                ds = dcmread(os.path.join(self.datFldrList[k]['dir'], f))
                # 2. Anonymize data
                anonymizer.anonymize(ds)
                # 3. Prepare output name
                outFileName = self.prepare_output_file_name(ds, self.datFldrList[k])
                # 4. Save anonymous DICOM in the output folder
                ds.save_as(os.path.join(outputDir, outFileName))
                
class AnonymizerGUI:
    def __init__(self, master):
        self.master = master
        master.title("DICOM Anonymizer")
        
        # Setting up the style and fonts
        style = ttk.Style()
        style.theme_use('clam')
        self.text_font = Font(family="Helvetica", size=10)
        
        # Configuration for the automatic resizing
        master.grid_columnconfigure(1, weight=1)
        
        # UI Elements
        self.setup_ui()

    def setup_ui(self):
        # Entry for Data Directory
        ttk.Label(self.master, text="Data Directory:", font=self.text_font).grid(row=1, sticky=tk.W, padx=10)
        self.data_dir_entry = ttk.Entry(self.master, font=self.text_font)
        self.data_dir_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(self.master, text="Browse", command=lambda: self.browse_dir('data')).grid(row=1, column=2, padx=5, pady=5)

        # Entry for Patient List Directory
        ttk.Label(self.master, text="Patient List Directory:", font=self.text_font).grid(row=2, sticky=tk.W, padx=10)
        self.pat_list_dir_entry = ttk.Entry(self.master, font=self.text_font)
        self.pat_list_dir_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(self.master, text="Browse", command=lambda: self.browse_dir('pat_list')).grid(row=2, column=2, padx=5, pady=5)

        # Entry for Starting Number
        ttk.Label(self.master, text="Starting File Number:", font=self.text_font).grid(row=3, sticky=tk.W, padx=10)
        self.start_number_entry = ttk.Entry(self.master, font=self.text_font)
        self.start_number_entry.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
        self.start_number_entry.insert(0, "1")  # Default value

        # Entry for Output Directory
        ttk.Label(self.master, text="Output Directory:", font=self.text_font).grid(row=4, sticky=tk.W, padx=10)
        self.output_dir_entry = ttk.Entry(self.master, font=self.text_font)
        self.output_dir_entry.grid(row=4, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(self.master, text="Browse", command=lambda: self.browse_dir('output')).grid(row=4, column=2, padx=5, pady=5)

        # Start Button
        ttk.Button(self.master, text="Start Anonymization", command=self.start_anonymization).grid(row=5, columnspan=3, padx=10, pady=10)
        ttk.Button(self.master, text="Organize Files", command=self.organize_files).grid(row=7, columnspan=3)

        # Status Box
        self.status_box = scrolledtext.ScrolledText(self.master, height=10, font=self.text_font)
        self.status_box.grid(row=6, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        self.status_box.config(state='disabled')
                
    def generate_patiet_aid_to_tid_map(self, pat_data_dir):
        patInfo = {'TID':[], 'AID':[]}
        patFldrs = {}
                
        for p in os.listdir(pat_data_dir):
            for root, dirs, files in os.walk(os.path.join(pat_data_dir, p)):
                for f in files:
                    try:
                        ds = dcmread(os.path.join(root, f))                        
                        if ds.data_element('Modality').value in ['CT', 'RTSTRUCT', 'RTPLAN', 'RTDOSE']:
                            patFldrs[p] = os.path.join(root, p)
                            break
                    except:
                        pass
                    
        if not self.start_number_entry.get().isdigit():
            messagebox.showwarning("Warning", "Please enter a valid integer for the starting number.")
            return 0
    
        vmin = int(self.start_number_entry.get())
        vmax = vmin + len(patFldrs)
        patInfo['TID'] = list(patFldrs.keys())
        patInfo['AID'] = np.random.choice(range(vmin, vmax), size=len(patInfo['TID']), replace=False).tolist()

        df = pd.DataFrame(patInfo)
        
        # check if output directory is specified
        if not self.output_dir_entry.get():
            self.pat_list_dir = os.path.join(pat_data_dir, "patient_tid_to_aid_mapping.csv")
        else:
            self.pat_list_dir = os.path.join(self.output_dir_entry.get(), "patient_tid_to_aid_mapping.csv")
            
        df.to_csv(self.pat_list_dir)
        
    def browse_dir(self, dir_type):
        folder_selected = filedialog.askdirectory()
        if dir_type == 'data':
            self.data_dir_entry.delete(0, tk.END)
            self.data_dir_entry.insert(0, folder_selected)
        elif dir_type == 'pat_list':
            self.pat_list_dir_entry.delete(0, tk.END)
            self.pat_list_dir_entry.insert(0, folder_selected)
        elif dir_type == 'output':
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, folder_selected)
            
    def organize_files(self):
        data_dir = self.data_dir_entry.get()
        output_dir = self.output_dir_entry.get()
        pat_list_dir = self.pat_list_dir_entry.get() if self.pat_list_dir_entry.get() else None
        
        if not data_dir:
            messagebox.showwarning("Warning", "Please specify the data directory")
            return

        self.update_status("Starting file organization...")
        
        # Initialize the DicomAnonymizer
        anonymizer = DicomAnonymizer(data_dir, pat_list_dir, output_dir)
        
        # Organize files
        # try:
        anonymizer.organize_dicom_files()
        self.update_status("File organization completed successfully.")
        organized_dir = os.path.join(output_dir or data_dir, 'Organized')
        self.update_status(f"Files organized in {organized_dir}")
        # except Exception as e:
            # self.update_status(f"An error occurred during file organization: {str(e)}")

    def start_anonymization(self):
        data_dir = self.data_dir_entry.get()
        pat_list_dir = self.pat_list_dir_entry.get()
        output_dir = self.output_dir_entry.get()
        
        if not pat_list_dir or pat_list_dir is None:
            error_code = self.generate_patiet_aid_to_tid_map(data_dir)
            pat_list_dir = self.pat_list_dir
            if error_code == 0:
                return
                        
        if not data_dir:
            messagebox.showwarning("Warning", "Please specify a data directory")
            return

        self.update_status("Starting anonymization...")
        # Initialize your anonymizer here
         
        anonymizer = DicomAnonymizer(data_dir, pat_list_dir)
        # Start the anonymization process (you might want to run this in a separate thread)
        # try:
        anonymizer.anonymize_dicom_files()
        self.update_status("Anonymization completed successfully.")
        output_dir = self.output_dir_entry.get() or os.path.join(data_dir, 'anonymized')
        self.update_status(f"Anonymized files saved in {output_dir}")
        # except Exception as e:
        #     self.update_status(f"An error occurred: {str(e)}")

    def update_status(self, message):
        self.status_box.config(state='normal')
        self.status_box.insert(tk.END, message + "\n")
        self.status_box.config(state='disabled')
        self.status_box.see(tk.END)

if __name__ == "__main__":    
    root = tk.Tk()
    root.geometry("600x400")  # Set initial size
    gui = AnonymizerGUI(root)
    root.mainloop()
