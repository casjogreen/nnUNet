import numpy as np

def siddon_ray_tracing(source_point, target_point, x, y, z):
    '''
    Python implementation of Siddon's ray tracing algorithm
    
    Created by Ivan Vazquez in collaboration with Ming Yang
    
    Parameters:
    `source_point::ndarray::(3,)` : Coordinates of the source given as an (x,y,z) triplet
    `target_point::ndarray::(3,)` : Coordinates of the target given as an (x,y,z) triplet
    `x::ndarray::(N,)` : X-coordinates of the planes perpendicular to the x-axis
    `y::ndarray::(N,)` : Y-coordinates of the planes perpendicular to the y-axis
    `x::ndarray::(N,)` : z-coordinates of the planes perpendicular to the z-axis
    
    Returns:
    `(i,j,k)::ndarray::(N,3)` Indices of voxels intersected by the ray
    `l::ndarray::(N,)` Lengths traversed by the ray inside all intersected voxels

    Reference:
    1. Siddon, R. L. Fast calculation of the exact radiological path for a three-dimensional CT array. Med Phys 12, 252–255 (1985).
    '''
    
    Nx, Ny, Nz = len(x)+1, len(y)+1, len(z)+1

    dx = np.abs(x[1]-x[0])
    dy = np.abs(y[1]-y[0])
    dz = np.abs(z[1]-z[0])
            
    # Add plane at end of coordinates
    xCoords = np.zeros(Nx)
    xCoords[0:-1], xCoords[-1] = x-dx/2, x[-1]+dx/2
    yCoords = np.zeros(Ny)
    yCoords[0:-1], yCoords[-1] = y-dy/2, y[-1]+dy/2
    zCoords = np.zeros(Nz)
    zCoords[0:-1], zCoords[-1] = z-dz/2, z[-1]+dz/2
        
    # compute useful variables
    X2_m_X1 = target_point[0] - source_point[0]
    Y2_m_Y1 = target_point[1] - source_point[1]
    Z2_m_Z1 = target_point[2] - source_point[2]
        
    # Calculate full ray length
    d12 = np.linalg.norm(source_point - target_point) 
    
    # 2. GET COORDINATES OF FIRST AND LAST PLANE
    X_plane_1, Y_plane_1, Z_plane_1 = xCoords[0], yCoords[0], zCoords[0] 
    X_plane_Nx, Y_plane_Ny, Z_plane_Nz = xCoords[-1], yCoords[-1], zCoords[-1]

    # 3. GET PARAMETRIC VALUES
    alpha_min_list, alpha_max_list = [0], [1]
    
    # x-plane
    if target_point[0] != source_point[0]:
        aX_1 = (X_plane_1 - source_point[0]) / X2_m_X1   
        aX_Nx = (X_plane_Nx - source_point[0]) / X2_m_X1
        alpha_min_list.append(np.min([aX_1, aX_Nx]))
        alpha_max_list.append(np.max([aX_1, aX_Nx]))
                
    # y-plane
    if target_point[1] != source_point[1]:
        aY_1 = (Y_plane_1 - source_point[1]) / Y2_m_Y1
        aY_Ny = (Y_plane_Ny - source_point[1]) / Y2_m_Y1
        alpha_min_list.append(np.min([aY_1, aY_Ny]))
        alpha_max_list.append(np.max([aY_1, aY_Ny])) 

    # z-plane
    if target_point[2] != source_point[2]:
        aZ_1 = (Z_plane_1 - source_point[2]) / Z2_m_Z1
        aZ_Nz = (Z_plane_Nz - source_point[2]) / Z2_m_Z1
        alpha_min_list.append(np.min([aZ_1, aZ_Nz]))
        alpha_max_list.append(np.max([aZ_1, aZ_Nz]))

    # 4. CALCULATE ALPHA_MIN AND ALPHA_MAX
    alpha_min = np.max([alpha_min_list])
    alpha_max = np.min([alpha_max_list])
    
    # 5. GET MIN AND MAX INDICES FOR EACH COORDINATE
    # x-planes
    if target_point[0] == source_point[0]:
        i_min = 0; i_max = 0
    elif target_point[0] > source_point[0]:
        i_min = int(Nx - (X_plane_Nx - alpha_min * X2_m_X1 - source_point[0])/dx)
        i_max = int(1 + (source_point[0] + alpha_max * X2_m_X1 - X_plane_1)/dx)
    else:
        i_min = int(Nx - (X_plane_Nx - alpha_max * X2_m_X1 - source_point[0])/dx)
        i_max = int(1 + (source_point[0] + alpha_min * X2_m_X1 - X_plane_1)/dx)

    # y-planes
    if target_point[1] == source_point[1]:
        j_min = 0; j_max = 0
    elif target_point[1] > source_point[1]:
        j_min = int(Ny - (Y_plane_Ny - alpha_min * Y2_m_Y1 - source_point[1])/dy)
        j_max = int(1 + (source_point[1] + alpha_max * Y2_m_Y1 - Y_plane_1)/dy)        
    else:
        j_min = int(Ny - (Y_plane_Ny - alpha_max * Y2_m_Y1 - source_point[1])/dy)
        j_max = int(1 + (source_point[1] + alpha_min * Y2_m_Y1 - Y_plane_1)/dy)
        
    # z-plane
    if target_point[2] == source_point[2]:
        k_min = 0; k_max = 0
    elif target_point[2] > source_point[2]:
        k_min = int(Nz - (Z_plane_Nz - alpha_min * Z2_m_Z1 - source_point[2])/dz) 
        k_max = int(1 + (source_point[2] + alpha_max * Z2_m_Z1 - Z_plane_1)/dz)
    else:
        k_min = int(Nz - (Z_plane_Nz - alpha_max * Z2_m_Z1 - source_point[2])/dz)
        k_max = int(1 + (source_point[2] + alpha_min * Z2_m_Z1 - Z_plane_1)/dz)
          
    # 6. FIND FINAL SET OF PARAMETRIC VALUES
    # x-direction
    if i_min != i_max:
        alpha_x =sorted([(xCoords[xInd]-source_point[0])/X2_m_X1 for xInd in np.arange(i_min, i_max)])
    else:
        alpha_x = [np.infty]
    
    # y-direction
    if j_min != j_max:
        alpha_y = sorted([(yCoords[yInd]-source_point[1])/Y2_m_Y1 for yInd in np.arange(j_min, j_max)])            
    else:
        alpha_y = [np.infty]
    
    # z-direction
    if k_min != k_max:
        alpha_z = sorted([(zCoords[zInd]-source_point[2])/Z2_m_Z1 for zInd in np.arange(k_min, k_max)])
    else:
        alpha_z = [np.infty]
    
    # 7. MERGE PARAMETRIC VALUESs    
    temp = np.unique(np.array([alpha_min]+[x for x in alpha_x]+[y for y in alpha_y]+[z for z in alpha_z] + [alpha_max]))
    alphas = temp[np.where(temp != np.infty)]
        
    # 8. CALCULATE VOXEL INTERSECTION LENGTHS
    l = d12 * np.diff(alphas)
            
    # 9. CALCULATE ALPHA_MID
    alphas_mid = .5*(alphas[0:-1] + alphas[1:])
    
    # 10. CALCULATE FINAL SET OF INDICES
    i_m = (source_point[0] + alphas_mid*X2_m_X1 - X_plane_1)/dx 
    j_m = (source_point[1] + alphas_mid*Y2_m_Y1 - Y_plane_1)/dy 
    k_m = (source_point[2] + alphas_mid*Z2_m_Z1 - Z_plane_1)/dz
        
    # round to get integer indices
    i = np.floor(i_m).astype(int)
    j = np.floor(j_m).astype(int)
    k = np.floor(k_m).astype(int)
    
    return (k,j,i), l