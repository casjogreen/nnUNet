import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py, os
import asyncio
from skimage import data, img_as_float
from skimage import exposure
import matplotlib as mpl
from additional_tools import compute_dvh
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
from user_config_interpreter import ConfigFileInterpreter

class Timer:
    def __init__(self, timeout, callback):
        self._timeout = timeout
        self._callback = callback

    async def _job(self):
        await asyncio.sleep(self._timeout)
        self._callback()

    def start(self):
        self._task = asyncio.ensure_future(self._job())

    def cancel(self):
        self._task.cancel()

def debounce(wait):
    """ Decorator that will postpone a function's
        execution until after `wait` seconds
        have elapsed since the last time it was invoked. """
    def decorator(fn):
        timer = None
        def debounced(*args, **kwargs):
            nonlocal timer
            def call_it():
                fn(*args, **kwargs)
            if timer is not None:
                timer.cancel()
            timer = Timer(wait, call_it)
            timer.start()
        return debounced
    return decorator

class VisualizationToolbox():
    
    def __init__(self, inference_data_directory):
        
        # identify patient files in the inference data directory
        pat_files = [f for f in os.listdir(inference_data_directory) if 'AID' in f]
        self.id_to_file_match = {p.split('-')[1].split('.')[0]:os.path.join(inference_data_directory,p) for p in pat_files}
        self.patient_ids = [str(p) for p in sorted([int(x) for x in self.id_to_file_match.keys()])]
        self.ct_im = None
        self.iso_im = None
        self.wash_im = None
        self.cont_im =None
        self.fig_size = (10,8)
        self.contour_linewidth = 3
        self.lgnd_inc = 6
        self.lgnd_init = 8
        self.__old_slice = 999999
        self.titles = ['Ground Truth', 'Predicted']
    
    def get_patient_data(self, patient_file_directory):
        
        with h5py.File(patient_file_directory,'r') as hf:
            gt_dose = hf['dose/ground_truth'][...]
            pr_dose = hf['dose/predicted'][...]
            ct = hf['ct'][...]
            rx_values = hf['rx_dose_values'][...]
            contours = {k:hf['contours'][k][...] for k in hf['contours'].keys()}
            (x,y,z) = hf['coordinates']['x'][...],hf['coordinates']['y'][...],hf['coordinates']['z'][...]
            
        self.ct_enhancement_status = False
        
        return gt_dose, pr_dose, ct, contours, rx_values, (x,y,z)
    
    def enhance_ct(self, ct, plane, clip_limit = 0.03):
        
        if plane == 'A':
            for n in range(ct.shape[0]):
                ct_slice = ct[n,:,:]
                if np.max(ct_slice) > 0: ct_slice /= np.max(ct_slice)
                ct_slice = exposure.equalize_adapthist(ct_slice, clip_limit=clip_limit)
                ct[n,:,:] = ct_slice

        elif plane == 'C':
            for n in range(ct.shape[1]):
                ct_slice = ct[:,n,:]
                if np.max(ct_slice) > 0: ct_slice /= np.max(ct_slice)
                ct_slice = exposure.equalize_adapthist(ct_slice, clip_limit=clip_limit)
                ct[:,n,:] = ct_slice

        elif plane == 'S':
            for n in range(ct.shape[2]):
                ct_slice = ct[:,:,n]
                if np.max(ct_slice) > 0: ct_slice /= np.max(ct_slice)
                ct_slice = exposure.equalize_adapthist(ct_slice, clip_limit=clip_limit)
                ct[:,:,n] = ct_slice
                
        self.ct_enhancement_status = True

        return ct
    
    def prepare_display_customization_widgets(self):
        
        # Checkbox to enable/disable isodose lines
        self.legend_box = widgets.Checkbox(value=True,
                                           description='Show Legend',
                                           disabled=False)

        # Checkbox to enable the display of one or more contours
        self.contour_box = widgets.Checkbox(value=False,
                                            description='Show Contours',
                                            disabled=False)

        # Checkbox to enable the display of one or more contours
        self.show_ct_box = widgets.Checkbox(value=True,
                                            description='Show CT',
                                            disabled=False)

        # Checkbox to enable the display of one or more contours
        self.enhance_ct_box = widgets.Checkbox(value=True,
                                               description='Enhance CT',
                                               disabled=False)

        # Radial buttons to select the desired view
        
        self.image_view = widgets.RadioButtons(options=[('Axial','A'), ('Coronal','C'), ('Sagittal','S')],
                                               value='S',
                                               description='View',
                                            #    layout={'height': '35%'}, 
                                               disabled=False)
        # self.image_view.layout = widgets.Layout(flex_flow='row')


        # Radial buttons to select the method of display for the dose 
        self.dose_view = widgets.RadioButtons(options=['Isodose Lines', 'Dose Wash', 'Both'],
                                              value='Isodose Lines', 
                                            #   layout={'height': '40%'}, 
                                              description='Dose',
                                              disabled=False)
        
        # self.dose_view.layout = widgets.Layout(flex_flow='row')

        
        # Alpha value for transparency of dose wash
        self.alpha = widgets.BoundedFloatText(value=0.5,
                                              min=0,
                                              max=1,
                                              step=0.1,
                                              layout={'width': '70%'}, 
                                              description='Alpha:',
                                              disabled=False)
        
        
        
        self.isodose_levels = widgets.Text(value='1.05, 1, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3',
                                           placeholder='Isodose Levels',
                                           description='Iso Levels:',
                                           disabled=False)
        self.isodose_levels.layout.width = '350px'
        
        self.initial_slice = widgets.Text(value='45',
                                          placeholder='Start at Slice',
                                          description='Initial Slice:',
                                          layout={'width': '32%'},
                                          disabled=False)
        # self.isodose_levels.layout.width = '350px'
        
        # Alpha value for transparency of dose wash
        self.min_dose_prcnt = widgets.BoundedFloatText(value=0.1,
                                                       min=0,
                                                       max=0.2,
                                                       step=0.01,
                                                       layout={'width': '32%'}, 
                                                       description='Dose TH:',
                                                       disabled=False)
        
        
        
        
        # Colormap to use when displaying the dose (wash or isodose lines)
        self.colormaps = widgets.Dropdown(options=['jet', 'rainbow','viridis', 'inferno', 'plasma','nipy_spectral','gist_rainbow','gnuplot'],
                                          value='jet',
                                          layout={'width': '70%'},
                                          description='Colormap:',)
        
    def setup_main_controls(self):
        
        # Left side of panel
        self.prepare_patient_selector()
        self.prepare_contour_options()
        lbox = widgets.VBox([self.selected_patient, self.contour_options]) # left box
        
        
        # middle and right
        self.prepare_display_customization_widgets()
        mbox = widgets.HBox([self.image_view, self.dose_view]) # middle box
        mbox = widgets.VBox([mbox, self.isodose_levels])
        
        mbox = widgets.VBox([mbox, widgets.HBox([self.initial_slice, self.min_dose_prcnt])])
        self.dose_view.layout.margin = '0 0 0 -100px'
        self.image_view.layout.margin = '0 0 0 25px'
        
        
        rbox = widgets.VBox([self.show_ct_box, self.enhance_ct_box, self.contour_box, self.legend_box, self.colormaps, self.alpha])
        
        self.image_view.observe(self.on_view_change, names='value')
        self.contour_box.observe(self.on_contour_button_change, names='value')
        self.contour_options.observe(self.on_contour_selections_change, names='value')
        
        combined = widgets.HBox([lbox, mbox, rbox])
        combined.layout.max_height = '700px'
        
        display(combined)
                
    def prepare_patient_selector(self):
        
        dd_options = [p for p in self.patient_ids]
        
        self.selected_patient = widgets.Dropdown(options=dd_options,
                                                 value=self.patient_ids[0],
                                                 description='Patient ID:',)
        
        outputs = self.get_patient_data(self.id_to_file_match[self.patient_ids[0]])
        self.gt_dose, self.pr_dose, self.ct, self.contours, self.rx_values, self.coords = outputs
        self.contour_names = [c for c in self.contours.keys()]
        
        self.selected_patient.observe(self.on_value_change, names='value')

    def prepare_contour_options(self, mode = 'multiple'):
        if mode != 'multiple':
            self.contour_options  = widgets.Select(options=self.contour_names,
                                                   value=self.contour_names[0],
                                                   rows=15,
                                                   description='Structures',
                                                   disabled=False)
        elif mode == 'multiple':
            self.contour_options  = widgets.SelectMultiple(options=self.contour_names,
                                                           value=[self.contour_names[0]],
                                                           rows=15,
                                                           description='Structures',
                                                           disabled=False)
        
    @debounce(0.1)
    def on_value_change(self,change):
        pat_id = change.new
        outputs = self.get_patient_data(self.id_to_file_match[pat_id])
        self.gt_dose, self.pr_dose, self.ct, self.contours, self.rx_values, self.coords = outputs
        self.contour_names = [c for c in self.contours.keys()]
        self.contour_options.options = self.contour_names
        self.contour_options.value = [self.contour_names[0]]
                
    def setup_slice_selector(self):
                
        if self.image_view.value == 'A':
            max_val = self.gt_dose.shape[0]
            
        elif self.image_view.value == 'S':
            max_val = self.gt_dose.shape[-1]
        
        elif self.image_view.value == 'C':
            max_val = self.gt_dose.shape[1]
                
        self.slice_selector = widgets.IntSlider(value=max_val//2,
                                                min=0,
                                                max=max_val,
                                                step=1,
                                                description='Slice:',
                                                disabled=False,
                                                continuous_update=False,
                                                orientation='horizontal',
                                                readout=True,
                                                readout_format='d')
        
        play = widgets.Play(value=max_val//2,
                            min=0,
                            max=max_val,
                            step=1,
                            interval=750,
                            description="Press play",
                            disabled=False)
        
        widgets.jslink((play, 'value'), (self.slice_selector, 'value'))
        fancy_slider = widgets.HBox([play, self.slice_selector])

        display(fancy_slider)
        
    def create_initial_plot(self, axis_off = True):
                        
        # Grab the name of the selected contours
        contour = self.contour_options.value
      
        # Get the slice number to display
        self.setup_slice_selector()
        sn = self.slice_selector.value
                
        # Enhance CT if needed
        if self.enhance_ct_box.value and not self.ct_enhancement_status: self.ct = self.enhance_ct(self.ct, self.image_view.value)
        
        # Grab x,y,z coords
        # x,y,z = self.coords
        z = np.abs(self.coords[0][1]-self.coords[0][0])*np.arange(self.gt_dose.shape[0])
        y = np.abs(self.coords[1][1]-self.coords[1][0])*np.arange(self.gt_dose.shape[1])
        x = np.abs(self.coords[2][1]-self.coords[2][0])*np.arange(self.gt_dose.shape[2])
        
        if self.image_view.value == 'A':
            extent = np.min(x),np.max(x),np.max(y),np.min(y)
            self.X, self.Y = x, y
            ctDat = self.ct[sn,:,:]
            gDose = self.gt_dose[sn,:,:]
            pDose = self.pr_dose[sn,:,:]
            if self.contour_box.value: 
                masks = {m:self.contours[m][sn,:,:].astype(int) for m in self.contour_options.value}
            self.lgndIndX = 0
            self.lgndIndY = 7
            self.sng = 1
            
        elif self.image_view.value == 'S': 
            extent = np.min(y),np.max(y), np.max(z),np.min(z)
            self.X, self.Y = y, z
            ctDat = self.ct[:,:,sn]
            gDose = self.gt_dose[:,:,sn]
            pDose = self.pr_dose[:,:,sn]
            if self.contour_box.value: 
                masks = {m:self.contours[m][:,:,sn].astype(int) for m in self.contour_options.value}
            self.lgndIndX = 0
            self.lgndIndY = -7
            self.sng = -1

        elif self.image_view.value == 'C': 
            extent = np.min(x),np.max(x), np.max(z), np.min(z)  
            self.X, self.Y = x, z 
            ctDat = self.ct[:,sn,:]
            gDose = self.gt_dose[:,sn,:]
            pDose = self.pr_dose[:,sn,:]
            if self.contour_box.value: 
                masks = {m:self.contours[m][:,sn,:].astype(int) for m in self.contour_options.value}
            self.lgndIndX = 0
            self.lgndIndY = -7
            self.sng = -1
            
        # Calculate isodose levels in Gy
        if self.dose_view.value in ['Isodose Lines', 'Both']:
            iso_levels = [float(n.strip()) for n in self.isodose_levels.value.split(',')]
            max_rx = np.max(self.rx_values)
            levels = [np.round(max_rx*f, 2) for f in iso_levels]
            levels+=[n for n in self.rx_values]
            self.iso_level_list = sorted(list(set([np.round(l,2) for l in levels]))) # remove repeated values
            
        # Prepare the dose data
        gDoseM = np.ma.masked_where(gDose <=  self.gt_dose.max()*self.min_dose_prcnt.value, gDose)
        pDoseM = np.ma.masked_where(pDose <=  self.pr_dose.max()*self.min_dose_prcnt.value, pDose)
        dose_data = [gDoseM, pDoseM]
        
        # Grab the minimum and maximum global dose values
        min_dose, max_dose = self.gt_dose.max()*self.min_dose_prcnt.value, self.gt_dose.max() # Define in terms of the global min and max

        # Grab colormap
        cmap = plt.get_cmap(self.colormaps.value)
        
        # Clear old figures
        plt.cla(),plt.clf(), plt.close('all')
        
        # Initialize variables used to contain images
        self.ct_im = {'gt':None, 'pr':None}
        self.iso_im = {'gt':None, 'pr':None}
        self.wash_im = {'gt':None, 'pr':None}
        self.cont_im = {'gt':{}, 'pr':{}}
    
        # Create a new figure (remove header and adjust spacing)
        fig, self.axes = plt.subplots(figsize =self.fig_size, nrows=1, ncols=2, frameon=False )
        fig.canvas.header_visible = False
        fig.canvas.toolbar_visible = False
        plt.subplots_adjust(wspace=0.07, hspace=0)
        
        # Get colors for isodose
        if self.dose_view.value in ['Isodose Lines', 'Both']:
            self.colors = [cmap(il/max_dose) for il in self.iso_level_list]
            self.rx_colors = [cmap(rx/max_dose) for rx in sorted(self.rx_values)]

        k = list(self.ct_im.keys())            
        for n, ax in enumerate(self.axes):

            if self.show_ct_box.value: 
                  self.ct_im[k[n]] = ax.imshow(ctDat, cmap='gray', extent=extent)

            if self.dose_view.value in ['Dose Wash', 'Both']:
                self.wash_im[k[n]] = ax.imshow(dose_data[n], extent=extent, vmin=min_dose, vmax=max_dose, 
                                               cmap = cmap, alpha=self.alpha.value, interpolation='bicubic')
                
            # Additional customizaton
            ax.set_title(self.titles[n], color='white')
            if self.image_view.value in ['C', 'S']: ax.invert_yaxis()
            if axis_off: ax.axis('off')

            # Isodose Plot
            if self.dose_view.value in ['Isodose Lines', 'Both']:
                self.iso_im[k[n]] = ax.contour(self.X, self.Y, dose_data[n], self.iso_level_list, colors=self.colors)
                for c in self.iso_im[k[n]].collections:
                    c.set_linewidth(self.contour_linewidth)
                    
            if self.contour_box.value:
                self.cont_im[k[n]] = {m:ax.contour(self.X, self.Y, masks[m], levels=1, colors=['w']) for m in masks}
                
            if self.dose_view.value in ['Isodose Lines', 'Both'] and self.legend_box.value:
                self.axes[0].text(self.X[self.lgndIndX], self.Y[self.lgndIndY], f'Isodose [cGy]', color='white', fontsize=10)
                for n,il in enumerate(self.iso_level_list[::-1]):        
                    self.axes[0].text(self.X[self.lgndIndX], self.Y[self.lgndIndY+self.sng*(self.lgnd_init+n*self.lgnd_inc)], f'{int(il*100)}', color=self.colors[::-1][n], fontsize=10)

        plt.tight_layout()
        
        self.slice_selector.observe(self.slice_value_change, names='value')
        
    def on_contour_button_change(self,change):
        self.update_plot(self.slice_selector.value)
            
    def on_contour_selections_change(self, change):
            self.update_plot(self.slice_selector.value)
        
    def on_view_change(self,change):
        
        outputs = self.get_patient_data(self.id_to_file_match[self.selected_patient.value])
        self.gt_dose, self.pr_dose, self.ct, self.contours, self.rx_values, self.coords = outputs
        self.ct = self.enhance_ct(self.ct, self.image_view.value)   
                
    def update_plot(self, sn):
        
        if self.image_view.value == 'A':
            ctDat = self.ct[sn,:,:]
            gDose = self.gt_dose[sn,:,:]
            pDose = self.pr_dose[sn,:,:]
            if self.contour_box.value: 
                masks = {m:self.contours[m][sn,:,:].astype(int) for m in self.contour_options.value}
            
        elif self.image_view.value == 'S':
            ctDat = self.ct[:,:,sn]
            gDose = self.gt_dose[:,:,sn]
            pDose = self.pr_dose[:,:,sn]
            if self.contour_box.value: 
                masks = {m:self.contours[m][:,:,sn].astype(int) for m in self.contour_options.value}

        elif self.image_view.value == 'C':
            ctDat = self.ct[:,sn,:]
            gDose = self.gt_dose[:,sn,:]
            pDose = self.pr_dose[:,sn,:]
            if self.contour_box.value: 
                masks = {m:self.contours[m][:,sn,:].astype(int) for m in self.contour_options.value}
        
        # Prepare the dose data
        gDoseM = np.ma.masked_where(gDose <= self.gt_dose.max()*self.min_dose_prcnt.value, gDose)
        pDoseM = np.ma.masked_where(pDose <= self.pr_dose.max()*self.min_dose_prcnt.value, pDose)
        dose_data = [gDoseM, pDoseM]
        
        if self.iso_im is None or self.wash_im is None: return
        
        # update ct and dose wash
        for n, k in enumerate(self.ct_im.keys()):
    
            # Update CT
            if self.show_ct_box.value:
                self.ct_im[k].set_data(ctDat)
                
            # Update dose wash
            if self.dose_view.value in ['Dose Wash', 'Both']:
                self.wash_im[k].set_data(dose_data[n])
                new_vmin = dose_data[n].min()
                new_vmax = dose_data[n].max()
                self.wash_im[k].set_clim(vmin=new_vmin, vmax=new_vmax)
                   
            # Update isodose    
            # Clear old values
            if self.iso_im[k] is not None:
                for coll in self.iso_im[k].collections:
                    coll.remove()
                self.iso_im[k] = None # reset variable
            
            if self.dose_view.value in ['Isodose Lines', 'Both']:    
                self.iso_im[k] = self.axes[n].contour(self.X, self.Y, dose_data[n], self.iso_level_list, colors=self.colors)
                for c in self.iso_im[k].collections:
                    c.set_linewidth(self.contour_linewidth)
                 
            # Contours
            # Clear any left over contours
            if self.cont_im[k] != {}:
                for m in self.cont_im[k].keys():
                    for coll in self.cont_im[k][m].collections:
                        coll.remove()
                self.cont_im[k] = {}
                                
            # Add new contours
            if self.contour_box.value: 
                for m in masks.keys():
                    self.cont_im[k][m] = self.axes[n].contour(self.X, self.Y,  masks[m], levels=1, colors=['w']) 
                                       
        plt.draw()
        
    @debounce(0.075)
    def slice_value_change(self, change):
        
        sn = change.new
        self.update_plot(sn)
        self.__old_slice = sn
        
    def show_dvh_comparison(self, mask_list = None, line_width = 2, fig_size = (9,6), include_targets = True, target_type = 'ctv'):

        cmap = plt.get_cmap('tab20')

        font = {'weight':'bold', 'size':13}
        
        mask_names ={
                    "body": "Body",
                    "brain": "Brain",
                    "brain_stem": "Brain Stem",
                    "oral_cavity": "Oral Cavity",
                    "chiasm": "Chiasm",
                    "left_cochlea": "Left Cochlea",
                    "right_cochlea": "Right Cochlea",
                    "esophagus": "Esophagus",
                    "left_lens": "Left Lens",
                    "right_lens": "Right Lens",
                    "left_eye": "Left Eye",
                    "right_eye": "Right Eye",
                    "larynx": "Larynx",
                    "left_submandibular_gland": "Left Submandibular Gland",
                    "right_submandibular_gland": "Right Submandibular Gland",
                    "mandible": "Mandible",
                    "left_optic_nerve": "Left Optic Nerve",
                    "right_optic_nerve": "Right Optic Nerve",
                    "left_parotid": "Left Parotid",
                    "right_parotid": "Right Parotid",
                    "spinal_cord": "Spinal Cord",
                    "rectum": "Rectum",
                    "bladder": "Bladder",
                    "femoral_heads": "Femoral Heads",
                    "heart": "Heart",
                    "left_lung": "Left Lung",
                    "right_lung": "Right Lung",
                    "thyroid_gland": "Thyroid",
                    "left_kidney": "Left Kidney",
                    "right_kidney": "Right Kidney",
                    "sacrum": "Sacrum",
                    "sigmoid": "Sigmoid",
                    "pelvic_bone": "Pelvis",
                    "left_breast": "Left Breast",
                    "right_breast": "Right Breast",
                    "small_bowel": "Small Bowel",
                    "large_bowel": "Large Bowel",
                    "duodenum": "Duodenum",
                    "liver": "Liver",
                    "stomach": "Stomach",
                    "spleen": "Spleen",
                    "left_brachial_plexus": "Left Brachial Plexus",
                    "right_brachial_plexus": "Right Brachial Plexus"
                    }

        # grab Rx dose list and generate target names
        mask_names = {**mask_names, **{f"{target_type}_{n.split('_')[-1]}":f"{target_type} {n.split('_')[-1]} Gy" for n in self.contours.keys() if target_type in n}}
        if mask_list is None: 
            mask_list = list(self.contours.keys())
        else:
            mask_list = [m for m in mask_list if m in self.contours.keys()] # ensure that the desired contours exist
            if include_targets: mask_list+= [m for m in self.contours if target_type in m and m not in mask_list] # add targets
            # mask_list = list(set(mask_list)) # remove repeats

        fig = plt.figure(figsize = fig_size)
        fig.canvas.header_visible = False
        fig.canvas.toolbar_visible = False
        ax = plt.subplot(111)
        d_max, c_num, legend_elements = 0, 0, []
        
        for k in mask_list:

            mask = self.contours[k]
            mask[np.where(mask>0)]= 1.0

            if not np.all(mask==0):
                
                

                if 'combined' not in k:

                    # ground truth plot
                    d,v = compute_dvh(mask, self.gt_dose)
                    ax.plot(d,v, c=cmap(c_num), linewidth=line_width, linestyle="-")
                    if np.max(d) > d_max: d_max = np.max(d)
                    
                    # dvh_data = {'dose': d, 'ground_truth':v}
                    # # save as csv using pandas
                    
                    # pd.DataFrame(dvh_data).to_csv(f"dvh_data/{k}_dvh_ground_truth.csv")

                    # predicted plot
                    d,v = compute_dvh(mask, self.pr_dose)
                    ax.plot(d, v, c=cmap(c_num), linewidth=line_width, linestyle="--",)
                    if np.max(d) > d_max: d_max = np.max(d)
                    
                    # dvh_data = {'dose': d, 'predicted':v}
                    # # save as csv using pandas
                    # pd.DataFrame(dvh_data).to_csv(f"dvh_data/{k}_dvh_predicted.csv")
                    
                    legend_elements.append(Patch(facecolor=cmap(c_num), label=mask_names[k]))

                    c_num += 1

        ax.set_ylim(0,None)
        ax.set_xlim(0,d_max)

        ax.set_xlabel('Dose [Gy]', fontdict=font)
        ax.set_ylabel('Volume [%]', fontdict=font)

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        legend_1 = plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1,1), fontsize=12)
        line_elements = [Line2D([0], [0], color='k', linestyle='-', lw=2, label='GT'), 
                         Line2D([0], [0], color='k', linestyle='--', lw=2, label='DL')]

        plt.gca().add_artist(legend_1)

        # plt.legend(handles=[line2], loc='lower right')
        plt.legend(handles=line_elements, loc='lower left', bbox_to_anchor=(1, 0), fontsize=12)

        plt.grid('on', linestyle='--', alpha=0.4)

def display_volume_slices(volume, cmap='gray'):
    """
    Display different slices of a volume using a slider widget.

    Parameters:
        volume (ndarray): 3D array representing the volume.
        cmap (str, optional): Colormap to be used for displaying the slices.
            Default is 'gray'.

    Returns:
        None

    Usage:
        # Assuming you have a 3D volume stored in the 'volume' variable
        display_volume_slices(volume, cmap='gray')
    """

    plt.cla(),plt.clf(), plt.close('all')

    min_val = volume.min()
    max_val = volume.max()
    
    # Define the slider widget
    slider = widgets.IntSlider(min=0, max=volume.shape[0]-1, value=0, description='Slice:')
    
    # Create the initial plot
    fig, ax = plt.subplots()
    im = ax.imshow(volume[slider.value, :, :], cmap=cmap, vmin=min_val, vmax=max_val)
    plt.axis('off')
    plt.show()

    # Define the update function
    def update_slice(change):
        # Update the displayed image with the selected slice
        im.set_array(volume[change.new, :, :])
        fig.canvas.draw_idle()

    # Attach the update function to the slider's value change event
    slider.observe(update_slice, names='value')

    # Display the slider
    display(slider)

            
class EvaluationFigures(ConfigFileInterpreter):
    
    def __init__(self, evaluation_results_dir, user_inputs_dir=None, expect_same_patients = False, **kwargs):
        
        super().__init__(user_inputs_dir)
        
        self.__logger = logging.getLogger(__name__)
        
        # check if the directory exists
        if not os.path.isdir(evaluation_results_dir):
            self.__logger.error(f"{evaluation_results_dir} is not a directory")
            raise ValueError(f"{evaluation_results_dir} is not a directory")
        
        # ensure that the directory is not empty
        eval_files = len(os.listdir(evaluation_results_dir))
        if eval_files == 0:
            self.__logger.error(f"{evaluation_results_dir} is empty")
            raise ValueError(f"{evaluation_results_dir} is empty")
        else:
            self.__logger.info(f"Found {eval_files} files in {evaluation_results_dir}")
            
        # read the evaluation result
        self.evaluation_results = {n.split(".")[0]:pickle.load(open(os.path.join(evaluation_results_dir, n), "rb")) for n in os.listdir(evaluation_results_dir)}

        # check if all of the elements in the list are the same
        pat_id_list = [sorted(list(self.evaluation_results[k].keys())) for k in self.evaluation_results.keys()]
        
        if not all([p == pat_id_list[0] for p in pat_id_list]):
            self.__logger.error(f"Patient IDs are not the same in all evaluation files.")
        if expect_same_patients: raise ValueError(f"Patient IDs are not the same in all files")          
        
        # Configure the plotting style
        if "font_scale" in kwargs.keys(): sns.set(font_scale=kwargs['font_scale'])
        sns.set_style("whitegrid")
                
        if "titles" in kwargs.keys():
            self.titles = kwargs['titles']
        else:
            self.titles = None
        
        if "include_mean" in kwargs.keys():
            self.include_average = kwargs['include_mean']
        else:
            self.include_average = False
        
        if "add_strip_plot" in kwargs.keys():
            self.strip_plot = kwargs['add_strip_plot']
        else:
            self.strip_plot = False
            
        if "boxplot_color" in kwargs.keys():
            self.boxplot_color = kwargs['boxplot_color']
        else:
            self.boxplot_color = "orange"
                        
    def get_oar_list(self):
        return list(self.user_inputs['DATA_PREPROCESSING']['oars']['name_and_voxel_value_pairs'].keys())
        
    def generate_boxplot(self, metrics, normalize=False, limit_pad=0.02):
        
        y_labels = {"gamma_passing_rate": "Gamma Passing Rate",
                    "dose_score": "Dose Score [Gy]",
                    "dvh_score": "DVH Score",
                    "homogeneity_index": "Difference",
                    "conformation_number": "Difference",
                    "d": "Dose Difference [Gy]",
                    "v": "Volume Difference [%]", 
                    "Dmax": "Dose Difference [Gy]",
                    "mean_dose": "Dose Difference [Gy]"}
        
        if normalize: 
            for m in  ['Dmax', 'Dmean', 'mean_dose', 'd', 'v']:
                y_labels[m] = "Percent of Prescription Dose" 
            self.normalize_by_prescription = True
        else:
            self.normalize_by_prescription = False
        
        # get the numbger of subplots
        n_plots = len(self.evaluation_results.keys())
        
        # create a figure for each metric
        for metric in metrics:
        
            if metric not in ['Dmax', 'Dmean', 'mean_dose']:
                fig, axs = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
            else:
                fig, axs = plt.subplots(n_plots, 1, figsize=(5*n_plots, 5*n_plots))
                            
            # get data limits
            data_limits = []
            
            for i, e in enumerate(self.evaluation_results.keys()):
                
                if metric in ['Dmax', 'Dmean', 'mean_dose']: i = n_plots-1-i
                                
                title = f"{e} {metric}" if self.titles is None else self.titles[i]
                
                pat_ids = sorted(list(self.evaluation_results[e].keys()))
                
                # get the data
                data, dat_lims = self.__get_data(self.evaluation_results[e], metric, pat_ids)
                     
                data_limits.append(dat_lims)
                                
                axis = axs[i] if n_plots > 1 else axs
                                
                if type(metric) == type(()) or type(metric) == type([]):
                    if all(['d' in m.lower() for m in metric]):
                        y_label = y_labels['d']
                    elif all(['v' in m.lower() for m in metric]):
                        y_label = y_labels['v']
                else:
                    y_label = y_labels[metric] 
                
                if metric in ['Dmax', 'Dmean', 'mean_dose']:
                    self.long_boxplot(data, ax=axis, ylabel=y_label, plot_number=n_plots-1-i, title=title) 
                else:
                    self.boxplot(data, ax=axis, ylabel=y_label, plot_number=i, title=title)
                    
            # add margins
            for ax in axs:
                if metric not in ['Dmax', 'Dmean', 'mean_dose']:
                    
                    data_range = np.max(data_limits) - np.min(data_limits)                    
                    ax.set_ylim([np.min(data_limits)-limit_pad*data_range, np.max(data_limits)+limit_pad*data_range])
                    
    def __get_data(self, evaluation_results, metric, pat_ids):
        
        if type(metric) == type(""): data = {metric:[]}
        
        if metric in ["homogeneity_index", "conformation_number"]:
            
            for p in pat_ids:
                targets = [k for k in evaluation_results[p].keys() if self.type_of_target_volume in k]
                for n in targets:
                    data[metric].append(evaluation_results[p][n][metric]['predicted']-evaluation_results[p][n][metric]['ground_truth']) 
                    
        if metric in ["Dmax", "Dmean", "mean_dose"]:
            
            organs = self.get_oar_list()
            data = {o:[] for o in organs}
            
            for p in pat_ids:
                rx_dose = evaluation_results[p]['rx_dose']
                
                for o in organs:
                    try:
                        diff = evaluation_results[p][o][metric]['predicted']-evaluation_results[p][o][metric]['ground_truth']       
                        if 100*self.normalize_by_prescription: diff = 100*np.abs(diff)/rx_dose
                        data[o].append(diff)
                    except:
                        pass
                                        
        elif metric in ["dose_score", "dvh_score", "gamma_passing_rate"]:
            
            data[metric] = [evaluation_results[p][metric] for p in pat_ids if p != 89]
            
            # remove values above 2
            # data[metric] = [d for d in data[metric] if d < 20]
            
        elif type(metric) == type(()) or type(metric) == type([]):
            
            data = {m:[] for m in metric}
                    
            if all(['d' in m.lower() for m in metric]):
                for p in pat_ids:
                    targets = [k for k in evaluation_results[p].keys() if self.type_of_target_volume in k]
                    rx_dose = evaluation_results[p]['rx_dose']
                    for n in targets:
                        for m in metric:
                            diff = evaluation_results[p][n][m]['predicted']-evaluation_results[p][n][m]['ground_truth']
                            if self.normalize_by_prescription: diff = 100*np.abs(diff)/rx_dose
                            data[m].append(diff)
                            
        # grab the maximum and minimum values in all of the data
        if metric not in ["Dmax", "mean_dose", "Dmean"]:
            data_limits = [np.min(data[m]) for m in data.keys() if data[m] != []] + [np.max(data[m]) for m in data.keys() if data[m] != []]
        else:
            data_limits = None
                                        
        return data, data_limits
            
    def boxplot(self, data, ax=None, ylabel=None, plot_number=0, title=None):
                
        showfliers = False if self.strip_plot else True
        
        # convert the data to a dataframe
        data = pd.DataFrame(data)
   
                
        sns.boxplot(data=pd.DataFrame(data),
                    notch=False, showcaps=True,
                    showfliers=False,
                    # flierprops={"marker": "x"},
                    boxprops={"facecolor": self.boxplot_color},
                    medianprops={"color": "k"}, 
                    ax=ax)
        
        if self.strip_plot:
            sns.stripplot(data=pd.DataFrame(data), jitter=0.2, size=4.5, ax=ax, alpha=0.5)
            
        ax.set_ylabel(ylabel)
        
        if len(data.keys()) == 1:
            labels = [item.get_text() for item in ax.get_xticklabels()]
            labels[0] = ''
            ax.set_xticklabels(labels)
        
        ax.set_title(title)

        # remove y-axis label if not the first plot
        if plot_number > 0:
            ax.set_ylabel('')
            ax.set_yticklabels('')
        
        if self.include_average and len(data.keys()) == 1:

            # Calculate statistics
            mean = np.mean(np.array(data[list(data.keys())[0]]))
            std = np.std(np.array(data[list(data.keys())[0]]))
            median = np.median(np.array(data[list(data.keys())[0]]))

            # Create legend handles and labels
            mean_std_patch = mpatches.Patch(color='none', label=r'$\mu \pm \sigma$: {:.2f} $\pm$ {:.2f}'.format(mean, std))
            median_patch = mpatches.Patch(color='none', label=f'Median: {median:.2f}')

            # Add legend to the plot with smaller font size
            ax.legend(handles=[mean_std_patch, median_patch], loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize='small', frameon=False)
            
    def long_boxplot(self, data, ax=None, ylabel=None, plot_number=0, title=None):
        
        keys = list(data.keys())
        
        if ylabel is None: ylabel = 'Dose Error [Gy]'

        df = pd.DataFrame({'OAR' : np.repeat(keys[0],len(data[keys[0]])), ylabel: data[keys[0]]})
                
        for o in list(data.keys()):
            new = pd.DataFrame({ 'OAR' : np.repeat(o,len(data[o])), ylabel: data[o] })
            df=pd.concat([df, new])
        
        ax.set_title(title)
        
        sns.boxplot(x='OAR', y=ylabel, data=df, ax=ax)
        # # add stripplot
        # ax = sns.stripplot(x='OAR', y='Dose Error [Gy]', data=df, color="orange", jitter=0.2, size=4.5)

        ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
        ax.set(xlabel=None);
        
        if plot_number > 0:
            labels = [item.get_text() for item in ax.get_xticklabels()]
            labels = ['' for l in labels]
            ax.set_xticklabels(labels)
                
    def dsc_plot(self, color = 'blueviolet', alpha=0.3, linewidth=1.5, generate_report = False, envelope_type='ci', inclue_all_curves = False):
        data = {}
        data_limits = []
        x = np.linspace(0.01,1,100)
        # get the numbger of subplots
        n_plots = len(self.evaluation_results.keys())
        
        fig, ax = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        
        for i, e in enumerate(self.evaluation_results.keys()):
            
            title = f"{e} DSC" 
            pat_ids = sorted(list(self.evaluation_results[e].keys()))
            pat_eval = self.evaluation_results[e]
            
            for p in pat_ids:
                
                rx_dose = pat_eval[p]['rx_dose']
                iso_vals = np.array(list(pat_eval[p]['DSC'].keys())).astype(float)
                dsc_vals = np.array([pat_eval[p]['DSC'][k] for k in iso_vals]).astype(float)
                data[p]= np.interp(x*rx_dose, iso_vals, dsc_vals)
                if inclue_all_curves:
                    ax[i].plot(x*100, data[p], color='blueviolet', alpha=0.3)
                # get data limits
                data_limits.append(np.min(data[p]))
                data_limits.append(np.max(data[p]))
            
            mean_dsc = np.mean([data[p] for p in data.keys()], axis=0)
            
            # get envelope
            if envelope_type is not None: 
                if envelope_type == 'ci':
                    envelope = 1.96*np.std([data[p] for p in data.keys()], axis=0)/np.sqrt(len(data.keys()))
                elif envelope_type == 'std':
                    envelope = np.std([data[p] for p in data.keys()], axis=0)
            
            
            ax[i].plot(x*100, mean_dsc, color='k', linewidth=3)
            ax[i].fill_between(x*100, mean_dsc-envelope, mean_dsc+envelope, color='k', alpha=0.2)
            
            ax[i].set_xlabel('Percentage of Prescription Dose')
            if i == 0:
                ax[i].set_ylabel('Dice Similarity Coefficient')
            else:
                ax[i].set_ylabel('')
                ax[i].set_yticklabels('')
        
        # find patient with highest mean deviation from the mean
        mean_deviation = {p:np.mean(np.abs(data[p]-mean_dsc)) for p in data.keys()}
        
        if generate_report:
            report = pd.DataFrame({'Patient ID':list(mean_deviation.keys()), 'Mean Deviation':list(mean_deviation.values())})
            # check if the directory exists
            if not os.path.isdir(os.path.join('temp','data')): os.mkdir(os.path.join('temp','data'))
            report.to_csv(os.path.join('temp','data','dsc_report.csv'))
               
        for i in range(len(ax)):
            data_range = np.max(data_limits) - np.min(data_limits)
            ax[i].set_ylim([np.min(data_limits)-0.02*data_range, 1])
            
        plt.subplots_adjust(wspace=0.05)