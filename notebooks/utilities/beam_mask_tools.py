import numpy as np
import logging
from scipy.ndimage import rotate as ndimage_rotate
from scipy.signal import convolve
from scipy.interpolate import RegularGridInterpolator
from siddon_ray_tracing import siddon_ray_tracing
import copy

class BeamMaskCreator():
    """Class to create a beam mask by tracing the beam path from the source to every point in a
    combined target volume.
    
    Created by Ivan Vazquez in collaboration with Ming Yang.
    
    Last updated: Spring 2024
    """
    
    def __init__(self, user_inputs) -> None:
        self.__logger = logging.getLogger(__name__)
        self.decimals = 4
        self.min_val = .01
        self.propagation_factor = 10
        self.user_inputs = user_inputs

    def create_beam_mask(self, dose, contours, plan, target_names):
        
        if not self.user_inputs["DATA_PREPROCESSING"]["beam_mask"]["include"]: return None
        
        # Grab values from user inputs
        kernel_size = self.user_inputs["DATA_PREPROCESSING"]["beam_mask"]["expansion_factor"]
        expand_target_margins = self.user_inputs["DATA_PREPROCESSING"]["beam_mask"]["expand_boundaries"] if kernel_size > 0 else False
                
        ## Combine targets volumes and initialize the beam mask
        combined_targets = np.zeros(contours[target_names[0]].data.shape)        
        for m in target_names:
            combined_targets += contours[m].data  
        combined_targets[np.where(combined_targets>0)] = 1
        beam_mask = np.zeros_like(combined_targets)
                
        # Expand the margins of the targets  
        if expand_target_margins:
            
            if self.user_inputs["DATA_PREPROCESSING"]["beam_mask"]["expansion_type"] == 'isotropic':
                self.__logger.info(f'Expanding target margins by {kernel_size} voxels isotropically')
                combined_targets = expand_contour_3d(combined_targets, fraction=kernel_size)
                
            else:
                
                expanded_targets = np.zeros_like(combined_targets)
            
                for bn in plan.beam.keys():
                    
                    isocenter = plan.beam[bn].isocenter
                    gantry_angle = plan.beam[bn].gantry_angle
                    couch_angle = plan.beam[bn].patient_support_angle
                    
                    if len(dose.keys()) > 1:  
                        coordinates = copy.deepcopy(dose[bn].coordinates.x), copy.deepcopy(dose[bn].coordinates.y), copy.deepcopy(dose[bn].coordinates.z)     
                    else:
                        dbn = list(dose.keys())[0]
                        coordinates = copy.deepcopy(dose[dbn].coordinates.x), copy.deepcopy(dose[dbn].coordinates.y), copy.deepcopy(dose[dbn].coordinates.z)                   
                    
                    expanded_targets+=self.__get_expanded_target_mask(kernel_size, combined_targets, coordinates, isocenter, gantry_angle, couch_angle)
            
            # update value of combined targets
            expanded_targets[np.where(expanded_targets > 0)] = 1
            combined_targets = copy.deepcopy(expanded_targets)
                             
        ## Get indices for the voxels inside of the targets
        inds = np.where(combined_targets > 0)
        
        # beam number for first beam in dose 
        dbn = list(dose.keys())[0]
        
        for bn in plan.beam.keys():
                                                                   
            # Grab the source to axis distance                        
            if dose[dbn].radiation_type == 'proton':
                SAD = np.mean(plan.beam[bn].vsad)
            else:
                SAD = plan.beam[bn].sad
                    
            # Grab the gantry and couch angles
            theta_g, theta_c = plan.beam[bn].gantry_angle, plan.beam[bn].patient_support_angle 
                                    
            # Grab isocenter location
            ix, iy, iz = plan.beam[bn].isocenter
            
            # Grab coordinates for the dose
            if len(dose.keys()) > 1:  
                x, y, z = copy.deepcopy(dose[bn].coordinates.x), copy.deepcopy(dose[bn].coordinates.y), copy.deepcopy(dose[bn].coordinates.z)    
            else:
                x, y, z = copy.deepcopy(dose[dbn].coordinates.x), copy.deepcopy(dose[dbn].coordinates.y), copy.deepcopy(dose[dbn].coordinates.z) 

            # Center the coordinates so that the isocenter is at (0,0,0)
            x -= ix
            y -= iy
            z -= iz
            
            # Create coordinate volumes
            zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
            
            # Use Siddon's Ray tracing algorithm to generate the beam mask for beam bn            
            sx, sy, sz = rotate_coordinates(np.array([0,-SAD,0]), theta_c, theta_g, precision = 4)
            
            for n in range(len(inds[0])):
            
                target = np.array([xx[inds[0][n], inds[1][n], inds[2][n]], yy[inds[0][n], inds[1][n], inds[2][n]], zz[inds[0][n], inds[1][n], inds[2][n]]])                
                source = np.array([sx, sy, sz]) 
                                
                if dose[dbn].radiation_type == 'photon':
                    target = self.__propagate_target_points_further(source, target)
                    
                locs, _ = siddon_ray_tracing(source, target, x, y, z) 
                
                if dose[dbn].radiation_type == 'photon':
                    locs = self.__check_indices(locs, beam_mask.shape)
                     
                if self.user_inputs["DATA_PREPROCESSING"]["beam_mask"]["binary"]:           
                    beam_mask[locs] = 1 
                else:
                    beam_mask[locs] += 1
                    
        maximum_value = self.user_inputs["DATA_PREPROCESSING"]["beam_mask"]["maximum_value"]
        
        if maximum_value is not None:
            if np.max(beam_mask) > 0:
                beam_mask = beam_mask/np.max(beam_mask)*maximum_value
            else:
                self.__logger.warning(f'Beam mask for patient {plan.patient_id} is empty. Maximum value not applied.')
                beam_mask = np.zeros_like(beam_mask)
         
        return beam_mask
       
    def __center_coordinates_at_isocenter(self, coords, iso):
            
        x,y,z = coords    
        
        x -= iso[0]
        y -= iso[1]
        z -= iso[2]
        
        return (x,y,z)
       
    def __get_expanded_target_mask(self, kernel_size, targets, coordinates, isocenter, gantry_angle, couch_angle):
        
        # Store original shape
        self.__original_shape = targets.shape
        
        # Set the center of the coordinats at the isocenter
        coordinates = self.__center_coordinates_at_isocenter(coordinates, isocenter)
        
        # Get BEV coordinates for the voxels in the original volume
        bev_coords = self.__get_bev_coordinates_for_data_volumes(coordinates, gantry_angle, couch_angle)
        
        # Rotate targets 
        rot_targets, rot_coords = self.__rotate_mask(targets, coordinates, couch_angle, gantry_angle)
        
        # Expand edges of contour
        kernel = np.ones((kernel_size,kernel_size))/kernel_size**2
        
        for j in range(rot_targets.shape[1]):
            rot_targets[:,j,:] = np.round(convolve(rot_targets[:,j,:], kernel, mode="same"), self.decimals)
        
        # Undo rotation
        expanded_targets = self.__undo_rotation(rot_targets, rot_coords, bev_coords, interp_method='linear')
        
        # Clean up contour and make is a binary mask
        expanded_targets = np.round(expanded_targets, self.decimals)
        expanded_targets[np.where(expanded_targets > 0)] = 1
        
        return expanded_targets
    
    def __get_bev_coordinates_for_data_volumes(self, coordinates, gantry_angle, couch_angle):
         
        # Transform Dose grid coordinates to BEV with 0 deg for both angles (couch and gantry) = BEV (0,0)
        y, z, x = coordinates
        
        # Save coordinate matrices for later use
        xx, zz, yy = np.meshgrid(x, z, y, indexing='ij')
        
        # Rotate coordinates
        coord_mat = np.stack((xx.flat, yy.flat, zz.flat), axis=1)
        
        x_r, y_r, z_r = self.__rotate_coordinates(coord_mat, couch_angle, gantry_angle) 
    
        return np.stack((x_r,z_r,y_r), axis=1)
    
    def __rotate_coordinates(self, coord_mat, theta_c, theta_g, decimals=4):
            
            # COUCH ROTATION
            theta = -theta_c # you need to rotate the coordinates by -theta to move them closer to the original system

            # rotation matrix for rotation about z-axis
            rot_mat = np.zeros([3,3])
            rot_mat[0,0], rot_mat[0,1] = np.cos(np.radians(theta)), np.sin(np.radians(theta))
            rot_mat[1,0], rot_mat[1,1] = -np.sin(np.radians(theta)), np.cos(np.radians(theta))
            rot_mat[2,2] = 1
            
            try:
                c_coords = np.round(np.dot(rot_mat, coord_mat.T), decimals)
            except:
                try:
                    c_coords = np.round(np.dot(rot_mat, coord_mat), decimals)
                except:
                    self.__logger.error("Error while applying the rotational matrix. Check dimensions")
                    raise ValueError("Error while applying the rotational matrix. Check dimensions")
                
            # GANTRY ROTATION
            theta = -theta_g
            
            # rotation matrix for rotations about the x-axis
            rot_mat = np.zeros([3,3])
            rot_mat[0,0] = 1
            rot_mat[1,1], rot_mat[1,2] = np.cos(np.radians(theta)), -np.sin(np.radians(theta))
            rot_mat[2,1], rot_mat[2,2] = np.sin(np.radians(theta)), np.cos(np.radians(theta))
            
            g_coords = np.round(np.dot(rot_mat,c_coords), decimals=4)
            
            # OUTPUT COORDINATES (x,y,z)
            return g_coords[0,:], g_coords[1,:], g_coords[2,:]
            
    def __rotate_mask(self, mask, coordinates, couch_angle, gantry_angle):
        
        x,y,z = coordinates
                    
        # 1 Couch rotation
        mask_cr = ndimage.rotate(mask, couch_angle, axes=(0,2), reshape=True, order = 1)
        
        # 1.1 Grab old coordinates along each axis of the (x,y) plane
        hCoords, sCoords, vCoords = coordinates
        
        # 1.2 Grab dimensions of rotated 
        rotShape = mask_cr[:,0,:].shape

        # 1.3 Update coordinates
        s, h, v = self.__get_rotated_coordinates(hCoords, vCoords, sCoords, rotShape, couch_angle) 
        x, y, z = h, s, v
        
        # 2 Gantry rotation
        mask_gr = ndimage.rotate(mask_cr, gantry_angle, axes=(1,2), reshape=True, order = 1)
        
        # 2.1 Update coordinates to BEV            
        vCoords, hCoords, sCoords = y, x, z
        rotShape = mask_gr[0,:,:].shape
        s, h, v = self.__get_rotated_coordinates(hCoords, vCoords, sCoords, rotShape, gantry_angle)
        x, y, z = s, h, v
          
        return mask_gr, (x,y,z)
    
    def __get_rotated_coordinates(self, hCoords, vCoords, sCoords, rotShape, angle, decimals = 4):
  
        if angle == 0: return sCoords, hCoords, vCoords    
    
        Nv, Nh = rotShape 

        h_o, v_o = self.__get_2D_coordinates_of_first_pixel(hCoords, vCoords, angle) 
                
        dh, dv = self.__get_voxel_pitch(angle, (Nv, Nh), hCoords, vCoords)
        
        v = np.round(np.array(v_o + np.arange(Nv) * dv), decimals)
        h = np.round(np.array(h_o + np.arange(Nh) * dh), decimals)
        s = np.round(sCoords,decimals)
        
        return s, h, v
    
    def __get_voxel_pitch(self, angle, dims, hCoords, vCoords):
        
        Nv, Nh = dims[0], dims[1]    
        Lh, Lv = hCoords.max() - hCoords.min(), vCoords.max() - vCoords.min()
            
        Lh_bev = np.abs(Lv*np.sin(np.radians(angle))) + np.abs(Lh*np.cos(np.radians(angle)))
        Lv_bev = np.abs(Lv*np.cos(np.radians(angle))) + np.abs(Lh*np.sin(np.radians(angle)))
        
        dv, dh = np.round(Lv_bev/(Nv-1), self.decimals), np.round(Lh_bev/(Nh-1), self.decimals)

        return dh, dv
    
    def __get_2D_coordinates_of_first_pixel(self, hCoords, vCoords, phi):
        
        # Get angle for each of the four corners of the system before rotation.
        rho_1 = np.linalg.norm((hCoords[0], vCoords[0]))
        theta_1 = np.rad2deg(np.arctan2(vCoords[0], hCoords[0])) - phi

        rho_2 = np.linalg.norm((hCoords[-1], vCoords[0]))
        theta_2 = np.rad2deg(np.arctan2(vCoords[0], hCoords[-1])) - phi

        rho_3 = np.linalg.norm((hCoords[-1], vCoords[-1]))
        theta_3 = np.rad2deg(np.arctan2(vCoords[-1], hCoords[-1])) - phi

        rho_4 = np.linalg.norm((hCoords[0], vCoords[-1]))
        theta_4 = np.rad2deg(np.arctan2(vCoords[-1], hCoords[0])) - phi

        # Patch to account for negative direction of couch rotations
        phiLoc = phi
        if phi < 0: phiLoc = 360 + phi

        if phiLoc <= 90:

            # get first x-coord from TL corner
            h = rho_1 * np.cos(np.radians(theta_1))
            v = rho_2 * np.sin(np.radians(theta_2))

        elif phiLoc > 90 and phiLoc <= 180:

            h = rho_2 * np.cos(np.radians(theta_2))
            v = rho_3 * np.sin(np.radians(theta_3))

        elif phiLoc > 180 and phiLoc <= 270:

            h = rho_3 * np.cos(np.radians(theta_3))
            v = rho_4 * np.sin(np.radians(theta_4))

        elif phiLoc > 270:

            h = rho_4 * np.cos(np.radians(theta_4))
            v = rho_1 * np.sin(np.radians(theta_1))

        return h, v
    
    def __propagate_target_points_further(self,source_point, target_point):
        x1, y1, z1 = source_point
        x2, y2, z2 = target_point
                
        # Calculate the direction vector
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        
        # Calculate the new point's coordinates
        x_new = x1 + self.propagation_factor * dx
        y_new = y1 + self.propagation_factor * dy
        z_new = z1 + self.propagation_factor * dz
        
        return (x_new, y_new, z_new)
    
    def __check_indices(self, indices, shape):
        
        indx = {'dim0': {0: [], 1: []}, 'dim1': {0: [], 1: []}, 'dim2': {0: [], 1: []}}
        dim_keys = ['dim0', 'dim1', 'dim2']  # map n to these keys
              
        for n, i in enumerate(indices):
            key = dim_keys[n]  # get the corresponding key

            if i[0] < 0:
                indx[key][0].append(np.argmax(i >= 0))
                
            elif i[0] > shape[n]:
                indx[key][0].append(np.argmax(i <= shape[n]-1))
          
            elif i[-1] < 0:
                indx[key][1].append(np.argmax(i < 0))        

            elif i[-1] > shape[n]:
                indx[key][1].append(np.argmax(i >= shape[n]))
                         
        first = max([x for k in indx.keys() for x in indx[k][0]]) if len([x for k in indx.keys() for x in indx[k][0]]) > 0 else 0
        last  = min([x for k in indx.keys() for x in indx[k][1]]) if len([x for k in indx.keys() for x in indx[k][1]]) > 0 else None
        
        dim0, dim1, dim2 = indices
        dim0 = dim0[first:last]
        dim1 = dim1[first:last]
        dim2 = dim2[first:last]
        return dim0, dim1, dim2
    
    def __undo_rotation(self, targets, rotated_coordinates, bev_coordinates, interp_method = 'linear'):
        
        # Grab coordinates of rotated dose distribution
        sR, hR, vR = rotated_coordinates # should be (x,y,z)

        # Generate interpolating function
        interpFunc = RegularGridInterpolator((sR, vR, hR), targets, bounds_error=False, 
                                             fill_value=0, method = interp_method) 
        
        # Get volume in original coordinate system
        return interpFunc(bev_coordinates).reshape(self.__original_shape)