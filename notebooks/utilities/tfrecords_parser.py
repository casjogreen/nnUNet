import tensorflow as tf

@tf.function
def generate_boolean_variables(probabilities):

    random_values = tf.random.uniform(shape=[len(probabilities)], dtype=tf.float32)
    boolean_variables = tf.less(random_values, probabilities)

    return boolean_variables

@tf.function
def apply_augmentations(x, types, flags, parameters):
    i = 0
    n = tf.size(flags)

    def cond(i, x, types, flags, parameters):
        return i < n

    def body(i,  x, types, flags, parameters):
        x = augmentation_operations(x, types[i], flags[i], parameters)
        return i + 1, x, types, flags, parameters

    x = tf.while_loop(cond, body, [i, x, types, flags, parameters])[1]
    return x

@tf.function
def augmentation_operations(data, aug, flag, parameters):   
    
    ## random horizontal flips
    cond_h_flip = tf.logical_and(tf.equal(aug, "h_flip"), flag)    
    data = tf.cond(cond_h_flip, lambda: tf.experimental.numpy.flip(data, axis=2), lambda: data)
    
    # random vertical flips
    cond_v_flip = tf.logical_and(tf.equal(aug, "v_flip"), flag)
    data = tf.cond(cond_v_flip, lambda: tf.experimental.numpy.flip(data, axis=1), lambda: data)
        
    return data

@tf.function
def get_structure_set(example_proto):
    
    # rebuild structure set from example_proto
    indices = tf.io.parse_tensor(example_proto['structures_indices'], out_type=tf.int64)
    values = tf.io.parse_tensor(example_proto['structures_values'], out_type=tf.float32)
    dense_shape = tf.io.parse_tensor(example_proto['structures_dense_shape'], out_type=tf.int64)
    
    structure_set = tf.sparse.SparseTensor(indices, values, dense_shape)
    
    # add structure set to data
    return tf.sparse.to_dense(structure_set)

@tf.function
def random_occlusion(inputs, targets, size, probability): 
            
    def false_fn(inputs):
        return inputs

    def true_fn(beam_mask, inputs, size):
        """Apply a random window to the beam mask of every element in a batch. 
        The window is applied based on the beam mask of the first element in the batch.
        """
                            
        # grab the beam mask for the first element in the batch
        beam_mask = beam_mask[:-size, :-size, :-size]        
        
        locations = tf.where(beam_mask > 0)
                
        # get the starting coordinates for the random window
        index = tf.random.uniform(shape=[], minval=0, maxval=tf.shape(locations)[0], dtype=tf.int32)        
        start_coords = locations[index]
                
        # Create a mask of ones initially
        mask = tf.ones_like(inputs[..., -1], dtype=tf.float32)
                
        # Get the starting coordinates
        x_start, y_start, z_start = start_coords[0], start_coords[1], start_coords[2]
                        
        # Generate indices where the mask will be zero
        indices = tf.reshape(tf.stack(tf.meshgrid(
            tf.range(x_start, x_start + size),
            tf.range(y_start, y_start + size),
            tf.range(z_start, z_start + size),
            indexing='ij'), axis=-1), (-1, 3))
        
        # Create the mask with zeros at the specified indices
        mask = tf.tensor_scatter_nd_update(mask, indices, tf.zeros(tf.shape(indices)[0]))
        
        # Create the final mask for all channels
        expanded_mask = tf.expand_dims(mask, axis=-1)
        final_mask = tf.concat([tf.ones_like(inputs[...,:-1]), expanded_mask], axis=-1)

        # plt.figure()
        # plt.imshow(final_mask[..., -1][x_start, :, :])
        # plt.imshow(inputs[..., -1][x_start, :, :], alpha=0.6)
                
        return tf.multiply(inputs, final_mask)
        
    # generate a flag to determine if the data should be augmented
    random_tensor = tf.random.uniform(shape=[], minval=0.0, maxval=1.0)    
    augment_data = tf.less(random_tensor, probability)
    inputs = tf.cond(tf.logical_and(tf.reduce_max(inputs[..., -1][:-size, :-size, :-size]) == 1.0, augment_data), 
                     lambda: true_fn(inputs[..., -1], inputs, size) , 
                     lambda: false_fn(inputs))
    return inputs, targets

@tf.function
def random_cropping(data, num_features, patch_loc, tlc, brc):
        
    # 1. Crop data for (dim_1, dim_2) planes    
    data = data[:, tlc[patch_loc,:][1]:brc[patch_loc,:][1], tlc[patch_loc,:][2]:brc[patch_loc,:][2]]
      
    # Split concatenated tensors and recombine to perform final cropping
    data = tf.split(data, num_or_size_splits=num_features, axis=0)
    data = tf.concat(data, axis = 2)
    
    # 2. Crop final dimensions
    data =  data[tlc[patch_loc,:][0]:brc[patch_loc,:][0], :, :]
    
    return data

@tf.function
def parse_training_tfrecords(example_proto, features_in_input, 
                             structures_in_target,include_voi_weights): 
                                                
    feature_description = {
        "ct" : tf.io.FixedLenFeature([], tf.string),
        "dose" : tf.io.FixedLenFeature([], tf.string),
        "target_volumes" : tf.io.FixedLenFeature([], tf.string),
        "beam_mask" : tf.io.FixedLenFeature([], tf.string),
        "model_inference" : tf.io.FixedLenFeature([], tf.string),
        "combined_oars" : tf.io.FixedLenFeature([], tf.string),
        "weights" : tf.io.FixedLenFeature([], tf.string),
        "structures_indices" : tf.io.FixedLenFeature([], tf.string),
        "structures_values" : tf.io.FixedLenFeature([], tf.string),
        "structures_dense_shape" : tf.io.FixedLenFeature([], tf.string),
        "structure_names" : tf.io.FixedLenFeature([], tf.string),
        "patch_size": tf.io.VarLenFeature(dtype=tf.int64),
        "top_left_corner": tf.io.FixedLenFeature([], tf.string),
        "bottom_right_corner": tf.io.FixedLenFeature([], tf.string),
        "patient_id": tf.io.FixedLenFeature([], tf.string),
        "type_of_patching": tf.io.FixedLenFeature([], tf.string),
        "dose_grid_size": tf.io.VarLenFeature(dtype=tf.int64),
        }
    
    # PARSE THE PROTO USING THE FEATURE DESCRIPTIONS 
    parsed_features = tf.io.parse_single_example(example_proto, feature_description) 
        
    ## parse variables needed for the following steps
    tlc = tf.io.parse_tensor(parsed_features["top_left_corner"], out_type=tf.int32)
    brc = tf.io.parse_tensor(parsed_features["bottom_right_corner"], out_type=tf.int32)
    dose_grid_size = tf.sparse.to_dense(parsed_features["dose_grid_size"], default_value=0)
    
    # PREPARE THE DATA ARRAY (INPUTS AND TARGETS)
    
    # get patient ID: for book keeping
    # pat_id = tf.strings.to_number(parsed_features['patient_id'], out_type=tf.int32)
        
    ## INPUT DATA
    ### Add CT
    data = tf.io.parse_tensor(parsed_features['ct'], out_type=tf.float32) 
    
    ### Add Target volume
    data = tf.concat([data, tf.io.parse_tensor(parsed_features['target_volumes'], out_type=tf.float32)], 0)   
    
    ### Add Combined OARs 
    include_combined_oars = tf.reduce_any(tf.equal(features_in_input, tf.constant("contours/combined_oars")))
    
    data = tf.cond(include_combined_oars, lambda: tf.concat([data, tf.io.parse_tensor(parsed_features['combined_oars'], out_type=tf.float32)], 0), 
                                          lambda: tf.concat([data, get_structure_set(parsed_features)], 0))
    
    ### Add Model inference
    include_beam_mask = tf.reduce_any(tf.equal(features_in_input, tf.constant("model_inference")))
    
    data = tf.cond(include_beam_mask, lambda: tf.concat([data, tf.io.parse_tensor(parsed_features['model_inference'], out_type=tf.float32)], 0), lambda: data)

    ### Add Beam mask    
    include_beam_mask = tf.reduce_any(tf.equal(features_in_input, tf.constant("contours/beam_mask")))   
    
    inputs = tf.cond(include_beam_mask, lambda: tf.concat([data, tf.io.parse_tensor(parsed_features['beam_mask'], out_type=tf.float32)], 0), lambda: data)

    ## TARGET DATA
    ### Add Dose
    data =  tf.io.parse_tensor(parsed_features['dose'], out_type=tf.float32) 
    
    #### Add the full structure set to the data (body + all other OARs)
    include_all_structures = tf.greater(tf.size(structures_in_target), tf.constant(1, dtype=tf.int32))
    
    data = tf.cond(include_all_structures, lambda: tf.concat([data, get_structure_set(parsed_features)], 0),
                                           lambda: data)
    
    #### Add the target volumes     
    include_target_volumes = tf.reduce_any(tf.equal(structures_in_target, tf.constant("contours/combined_targets")))
    
    data = tf.cond(include_target_volumes, lambda: tf.concat([data, tf.io.parse_tensor(parsed_features['target_volumes'], out_type=tf.float32)], 0), 
                                           lambda: data)
    
    #### Add beam mask
    include_beam_mask = tf.reduce_any(tf.equal(structures_in_target, tf.constant("contours/beam_mask")))
    
    data = tf.cond(include_beam_mask, lambda: tf.concat([data, tf.io.parse_tensor(parsed_features['beam_mask'], out_type=tf.float32)], 0), 
                                      lambda: data)
    
    #### Add only the body 
    include_only_body = tf.equal(tf.size(structures_in_target), tf.constant(1, dtype=tf.int32))
    
    data = tf.cond(include_only_body, lambda: tf.concat([data, get_structure_set(parsed_features)[:dose_grid_size[0], :, :]], 0), 
                                      lambda: data)
    
    ### Add Weights
    targets = tf.cond(include_voi_weights, lambda: tf.concat([data, tf.io.parse_tensor(parsed_features['weights'], out_type=tf.float32)], 0), lambda: data)
    
    return inputs, targets, tlc, brc

@tf.function
def augment_dataset(inputs, targets, tlc, brc, input_size, target_size, aug_parameters):
    
    # PREPARE STOCHASTIC QUANTITIES
    patch_loc = tf.random.uniform(shape=[], minval = 0, maxval=tf.shape(tlc)[0], dtype=tf.int32)
    aug_types = tf.random.shuffle(aug_parameters.types)
    aug_flags = generate_boolean_variables(aug_parameters.probabilities)
    
    # INPUT DATA
    ## Apply augmentation  
    inputs = apply_augmentations(inputs, aug_types, aug_flags, aug_parameters) 
    
    ## Apply random cropping    
    inputs = random_cropping(inputs, input_size, patch_loc, tlc, brc)
        
    ## Split data and recombine (stack) to finish preparing inputs
    inputs = tf.split(inputs, num_or_size_splits=input_size, axis=-1)
    inputs = tf.stack(inputs, axis=-1)
    
    # TARGET DATA
    ## Apply augmentation
    targets = apply_augmentations(targets, aug_types, aug_flags, aug_parameters)   
    
    ## Apply random cropping    
    targets = random_cropping(targets, target_size, patch_loc, tlc, brc)

    ## Split data and recombine (stack) to finish preparing inputs
    targets = tf.split(targets, num_or_size_splits=target_size, axis=-1)
    targets = tf.stack(targets, axis=-1)
    
    return inputs, targets
    
@tf.function
def parse_validation_tfrecords(example_proto, input_size, target_size, 
                               features_in_input, structures_in_target, 
                               include_voi_weights):

    feature_description = {
            "ct" : tf.io.FixedLenFeature([], tf.string),
            "dose" : tf.io.FixedLenFeature([], tf.string),
            "target_volumes" : tf.io.FixedLenFeature([], tf.string),
            "beam_mask" : tf.io.FixedLenFeature([], tf.string),
            "model_inference" : tf.io.FixedLenFeature([], tf.string),
            "combined_oars" : tf.io.FixedLenFeature([], tf.string),
            "weights" : tf.io.FixedLenFeature([], tf.string),
            "structures_indices" : tf.io.FixedLenFeature([], tf.string),
            "structures_values" : tf.io.FixedLenFeature([], tf.string),
            "structures_dense_shape" : tf.io.FixedLenFeature([], tf.string),
            "structure_names" : tf.io.FixedLenFeature([], tf.string),
            "patient_id": tf.io.FixedLenFeature([], tf.string),
            "dose_grid_size": tf.io.VarLenFeature(dtype=tf.int64),}
    
    # PARSE THE PROTO USING THE FEATURE DESCRIPTIONS 
    parsed_features = tf.io.parse_single_example(example_proto, feature_description) 
    
    ## parse variables needed for the following steps
    dose_grid_size = tf.sparse.to_dense(parsed_features["dose_grid_size"], default_value=0)
           
    # PREPARE THE DATA ARRAY (INPUTS AND TARGETS)
    
    ## INPUT DATA
    ### Add CT
    data = tf.io.parse_tensor(parsed_features['ct'], out_type=tf.float32) 

    ### Add Target volume
    data = tf.concat([data, tf.io.parse_tensor(parsed_features['target_volumes'], out_type=tf.float32)], 0)   
    
    ### Add Combined OARs 
    include_combined_oars = tf.reduce_any(tf.strings.regex_full_match(features_in_input, "contours/combined_oars"))
    
    data = tf.cond(include_combined_oars, lambda: tf.concat([data, tf.io.parse_tensor(parsed_features['combined_oars'], out_type=tf.float32)], 0), 
                                          lambda: tf.concat([data, get_structure_set(parsed_features)], 0))
    
    ### Add Model inference
    include_beam_mask = tf.reduce_any(tf.strings.regex_full_match(features_in_input, "model_inference"))
    
    data = tf.cond(include_beam_mask, lambda: tf.concat([data, tf.io.parse_tensor(parsed_features['model_inference'], out_type=tf.float32)], 0), lambda: data)

    ### Add Beam mask
    include_beam_mask = tf.reduce_any(tf.strings.regex_full_match(features_in_input, "contours/beam_mask"))
    
    data = tf.cond(include_beam_mask, lambda: tf.concat([data, tf.io.parse_tensor(parsed_features['beam_mask'], out_type=tf.float32)], 0), lambda: data)
    
    ### Split data and recombine (stack) to finish preparing inputs
    data = tf.split(data, num_or_size_splits=input_size, axis=0)
    inputs = tf.stack(data, axis=-1)
        
    ## TARGET DATA
    ### Add Dose
    data =  tf.io.parse_tensor(parsed_features['dose'], out_type=tf.float32) 
    
    ### Add Structures

    #### Add the full structure set to the data (body + all other OARs)
    include_all_structures = tf.greater(tf.size(structures_in_target), tf.constant(1, dtype=tf.int32))
    
    data = tf.cond(include_all_structures, lambda: tf.concat([data, get_structure_set(parsed_features)], 0),
                                           lambda: data)
    
    #### Add the target volumes 
    include_target_volumes = tf.reduce_any(tf.strings.regex_full_match(structures_in_target, "contours/combined_targets"))
    
    data = tf.cond(include_target_volumes, lambda: tf.concat([data, tf.io.parse_tensor(parsed_features['target_volumes'], out_type=tf.float32)], 0), 
                                           lambda: data)
    
    #### Add beam mask
    include_beam_mask = tf.reduce_any(tf.strings.regex_full_match(structures_in_target, "contours/beam_mask"))
    
    data = tf.cond(include_beam_mask, lambda: tf.concat([data, tf.io.parse_tensor(parsed_features['beam_mask'], out_type=tf.float32)], 0), 
                                      lambda: data)
    
    #### Add only the body 
    include_only_body = tf.equal(tf.size(structures_in_target), tf.constant(1, dtype=tf.int32))
        
    data = tf.cond(include_only_body, lambda: tf.concat([data, get_structure_set(parsed_features)[:dose_grid_size[0], :, :]], 0), 
                                      lambda: data)
    
    ### Add Weights
    data = tf.cond(include_voi_weights, lambda: tf.concat([data, tf.io.parse_tensor(parsed_features['weights'], out_type=tf.float32)], 0), lambda: data)
    
    ### Split data and recombine (stack) to finish preparing inputs
    data = tf.split(data, num_or_size_splits=target_size, axis=0)
    targets = tf.stack(data, axis=-1)
        
    return inputs, targets