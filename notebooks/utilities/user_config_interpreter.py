import json, logging, datetime, h5py, os
import tensorflow as tf
from collections import namedtuple

DataAugmentation = namedtuple('DataAugmentation', ['types', 'probabilities'])

class ConfigFileInterpreter():
    
    # TODO: Capture errors like instances when the preprocessed data is missing but the user is not performing preprocessing before training
    
    def __init__(self, user_input_directory, log_type = ""):
        
        self.__logger = logging.getLogger(__name__)
         
        if user_input_directory is None: 
            self.__logger.warning("No user input directory specified. Please specify a directory containing a JSON file with user parameters.")
            return
    
        self.up_dir = user_input_directory   
        self.__log_type = log_type
        self.read_user_parameters
        
    def produce_training_tags(self):    
        values = [self.model_name]
        values.append(datetime.datetime.now().strftime("%m/%d/%Y"))
        values.append('x'.join([str(x) for x in self.patch_dimensions]))
        with h5py.File(self.training_data_dir,'r') as hf:
            tot_num_pats = len(hf.keys())
            values.append(tot_num_pats)

        values.append(f'"{self.train_split},{tot_num_pats-self.train_split+self.test_split},{self.test_split}"')
        values.append(self.batch_size)
        if tot_num_pats > 300: values.append('Bilateral')
        if tot_num_pats < 300: values.append('Unilateral')
        values+=["0"]*3
        nrm_strat_ct = self.user_inputs["DATA_PREPROCESSING"]["ct"]["normalization_strategy"]
        nrm_strat_dose = self.user_inputs["DATA_PREPROCESSING"]["dose"]["normalization_strategy"]
        nrm__strat_oars = self.user_inputs["DATA_PREPROCESSING"]["oars"]["normalization_strategy"]
        values.append(f'"{nrm_strat_dose},{nrm_strat_ct},{nrm__strat_oars}"')

        values.append('"{}"'.format(','.join(self.loss_types)))
        res = self.user_inputs["DATA_PREPROCESSING"]["data_resolution"]["constant_voxel_dimensions"]
        if self.user_inputs["DATA_PREPROCESSING"]["data_resolution"]["use_constant_resolution"]:
            values.append(f'"{res[0]},{res[1]},{res[2]}"')
        else:
            values.append("original dose resolution")
        values.append(self.user_inputs["DATA_PREPROCESSING"]["oars"]["contour_set"])
        values.append(self.user_inputs["DATA_PREPROCESSING"]["beam_mask"]["include"])
        values.append(self.user_inputs["DATA_PREPROCESSING"]["beam_mask"]["expand_boundaries"])
        values.append(self.user_inputs["DATA_PREPROCESSING"]["oars"]["combine_volumes"])
        values.append("-")
        values.append(self.lr_decay["type"])
        values+=["0"]*3
        values.append('"{}"'.format(','.join(self.augmentation_types)))
        values.append(self.optimizer["type"])
        values.append('-')
        values.append(self.initial_learning_rate)

        # save training tag with date and time
        datetime_str = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        with open(os.path.join('logs', f'training_tags_{datetime_str}.csv'), 'w') as f:
            f.write(','.join([str(x) for x in values]))

    @property
    def read_user_parameters(self):
        """ Reads user parameters from a JSON file and stores them as instance variables.

            This method reads the user parameters from the JSON file specified in the constructor
            and assigns the relevant values to various instance variables of the JsonWrangler object.
            The JSON file should follow a specific structure containing directories, preprocessing settings,
            training parameters, evaluation parameters, and other configuration options.

            Example:
            --------
            json_wrangler = JsonWrangler(user_input_directory, write_logs=True)
            json_wrangler.read_user_parameters()

            Notes:
            ------
            - The JSON file should exist and be accessible at the specified user_input_directory.
            - The JSON file should follow a specific structure to ensure correct assignment of parameters.
            - After calling this method, the user parameters can be accessed as instance variables of the JsonWrangler object.

            """
        
        with open(self.up_dir, "r") as f:
            user_inputs = json.load(f)
        self.user_inputs = user_inputs

        # DIRECTORIES
        self.patient_data_dir = user_inputs["DIRECTORIES"]["raw_patient_data"]
        self.training_data_dir = user_inputs["DIRECTORIES"]["preprocessed_patient_data"]
        self.data_split_info_dir = user_inputs["DIRECTORIES"]["data_split"]
        self.weights_dir = user_inputs["DIRECTORIES"]["model_weights"] 
        self.model_inferences_dir = user_inputs["DIRECTORIES"]["model_inference"] 

        self.number_of_folds = user_inputs["INPUT_PIPELINE"]["number_of_folds"]   
        self.data_stratification = user_inputs['INPUT_PIPELINE']['data_stratification']
        self.train_split = user_inputs["INPUT_PIPELINE"]["train_split"]  
        self.test_split = user_inputs["INPUT_PIPELINE"]["test_split"]  
        self.write_new_records = user_inputs["INPUT_PIPELINE"]["write_new_records"]  
        self.load_data_split = user_inputs["INPUT_PIPELINE"]["load_data_split"]  
        self.patients_per_tfrecord = user_inputs["INPUT_PIPELINE"]["patient_per_tfrecord"] 
        self.shuffle_buffer_size = user_inputs["INPUT_PIPELINE"]["shuffle_buffer_size"]    
        self.seed = user_inputs["INPUT_PIPELINE"]["seed"] 
        self.model_name = user_inputs["MODEL"].lower()
        self.type_of_target_volume = user_inputs["TYPE_OF_TARGET_VOLUME"].lower()
        
        self.patch_type = user_inputs["DATA_PREPROCESSING"]["patches"]["type"]
        self.patch_dimensions = user_inputs["DATA_PREPROCESSING"]["patches"]["patch_dimensions"]
        self.patch_stride = user_inputs["DATA_PREPROCESSING"]["patches"]["patch_stride"]    
          
        # TRAINING PARAMETERS
        self.k_fold = str(user_inputs["TRAINING_PARAMETERS"]["k_fold"])
        self.batch_size = user_inputs["TRAINING_PARAMETERS"]["batch_size"] 
        self.augmentation_types = user_inputs["TRAINING_PARAMETERS"]["data_augmentation"]["types"]
        self.augmentation_parameters = user_inputs["TRAINING_PARAMETERS"]["data_augmentation"]["parameters"]
        self.cache_validation_dataset = user_inputs["TRAINING_PARAMETERS"]["cache_validation_dataset"]
        self.cache_training_dataset = user_inputs["TRAINING_PARAMETERS"]["cache_training_dataset"]
        self.augment_validation_dataset = user_inputs["TRAINING_PARAMETERS"]["data_augmentation"]["augment_validation_dataset"]
        self.epochs = user_inputs["TRAINING_PARAMETERS"]["epochs"]    
        self.loss_function = user_inputs["TRAINING_PARAMETERS"]["loss_function"]   
        self.dataset_repeats = {'training-set':user_inputs["TRAINING_PARAMETERS"]["training_dataset_repeats"]}  
        self.dataset_repeats['validation-set'] = user_inputs["TRAINING_PARAMETERS"]["validation_dataset_repeats"] 
        self.dataset_repeats['test-set'] = 1
        self.initial_learning_rate = user_inputs["TRAINING_PARAMETERS"]["initial_learning_rate"]               
        self.patience_es = user_inputs["TRAINING_PARAMETERS"]["patience_es"]   
        self.optimizer = user_inputs["TRAINING_PARAMETERS"]["optimizer"]  
        self.use_early_stopping = user_inputs["TRAINING_PARAMETERS"]["use_early_stopping"]  
        # Callbacks
        self.lr_decay = user_inputs["TRAINING_PARAMETERS"]["learning_rate_decay"]
        self.lr_decay["type"] = self.lr_decay["type"].lower()
        self.save_best_weights = user_inputs["TRAINING_PARAMETERS"]["callbacks"]["save_best_weights"]  
        self.save_weights_for_every_epoch = user_inputs["TRAINING_PARAMETERS"]["callbacks"]["save_weights_for_every_epoch"] 
        self.enable_tensorboard = user_inputs["TRAINING_PARAMETERS"]["callbacks"]["enable_tensorboard"]
        self.activate_profiler = user_inputs["TRAINING_PARAMETERS"]["callbacks"]["activate_profiler"]
        self.verbose = user_inputs["TRAINING_PARAMETERS"]["verbose"]
        self.memory_usage_callback = user_inputs["TRAINING_PARAMETERS"]["callbacks"]["memory_usage_logger"]
        
        # EVALUATION  
        self.data_set_to_evaluate = user_inputs["EVALUATION_PARAMETERS"]["data_set_to_evaluate"]  
        self.rim_crop = user_inputs["EVALUATION_PARAMETERS"]["rim_crop"]
        self.dta = user_inputs["EVALUATION_PARAMETERS"]["dta"]  
        self.dd = user_inputs["EVALUATION_PARAMETERS"]["dd"]  
        self.dd_method = user_inputs["EVALUATION_PARAMETERS"]["dd_method"]  
        self.dose_threshold = user_inputs["EVALUATION_PARAMETERS"]["dose_threshold"] 
        self.refinement_ratio = user_inputs["EVALUATION_PARAMETERS"]["refinement_ratio"]  
        self.parallelize_evaluation = user_inputs["PARALLELIZATION"]["parallelize_evaluation"]  
        self.eval_folds = user_inputs["EVALUATION_PARAMETERS"]["folds"]  
        self.eval_epoch = user_inputs["EVALUATION_PARAMETERS"]["epoch"]
        self.apply_body_mask = user_inputs["EVALUATION_PARAMETERS"]["apply_body_mask"]  
        self.min_dose = user_inputs["EVALUATION_PARAMETERS"]["min_dose"]  
        self.eval_patch_dimensions = user_inputs["EVALUATION_PARAMETERS"]["patch_dimensions"]
        self.eval_patch_stride = user_inputs["EVALUATION_PARAMETERS"]["patch_stride"]
        self.eval_augmentation = user_inputs["EVALUATION_PARAMETERS"]["augmentation"]
        self.prediction_file_mode = user_inputs["EVALUATION_PARAMETERS"]["prediction_file_mode"]  
        self.vol_for_max_dose = user_inputs["EVALUATION_PARAMETERS"]["vol_for_max_dose_in_cc"] 
        self.normalization_method = user_inputs["EVALUATION_PARAMETERS"]["normalization"]["type"]
        self.reference_normalization_volume = user_inputs["EVALUATION_PARAMETERS"]["normalization"]["reference_volume"]
        self.include_gpr_analysis = user_inputs["EVALUATION_PARAMETERS"]["include_gpr_analysis"]
        
        # handle the data augmentation parameters
        augmentation_types = tf.constant([x for x in self.augmentation_types if x != 'random_occlusion'], dtype=tf.string)
        probabilities = [self.augmentation_parameters[a]['probability'] for a in self.augmentation_types if 'occlusion' not in a]
        probabilities = tf.constant(probabilities, dtype=tf.float32)
        
        # shift_axes = tf.constant(self.augmentation_parameters['random_translation']['shift_axes'], dtype=tf.bool)
        self.augmentation_details = DataAugmentation(augmentation_types, probabilities)

        # get the types of loss functions that will be used
        self.loss_types = [v['type'].lower() for k,v in self.loss_function.items() if 'term' in k.lower() and v['type'] is not None and v["weight"] > 0]
                
        # set weight matrix flag to true to simplify future updates
        if any(['weighted' in x for x in self.loss_types]):
            self.use_weight_matrix = True
        else:
            self.use_weight_matrix = False
            
        # Determine the names of the input volumes
        self.inputs = ['ct']
        if user_inputs["DATA_PREPROCESSING"]["targets"]["combine_volumes"]: self.inputs+=['contours/combined_targets']
     
        if user_inputs["DATA_PREPROCESSING"]["oars"]["combine_volumes"]:
            self.inputs += ['contours/combined_oars']
            if user_inputs["DATA_PREPROCESSING"]["oars"]["separate_body_channel"]: 
                self.inputs += ['contours/body']    
        else:
            self.inputs += [f'contours/{n}' for n in user_inputs["DATA_PREPROCESSING"]["oars"]["name_and_voxel_value_pairs"].keys()]

        if user_inputs["DATA_PREPROCESSING"]["beam_mask"]["include"]: self.inputs+=['contours/beam_mask']
        
        if self.user_inputs["DATA_PREPROCESSING"]["include_model_inference"]: self.inputs+=['model_inference']

        # Prepare the names of the target volumes to use
        self.targets = ['dose']
        
        if self.use_weight_matrix: 
  
            if any(['dvh' in x for x in self.loss_types]):
                oars = [f'contours/{n}' for n in user_inputs["DATA_PREPROCESSING"]["oars"]["name_and_voxel_value_pairs"].keys()]
                self.targets = self.targets + ["contours/body"] + ["contours/combined_targets"] + [x for x in oars if x != "contours/body"]
                
            if any(['focused' in x for x in self.loss_types]): 
                if 'contours/body' not in self.targets: 
                    self.targets += ["contours/body"]
                else:
                    self.targets = self.targets + ["contours/body"] + [x for x in self.targets if x not in ["dose", "contours/body"]]
                       
            self.targets += ["weights"]
        
        elif any(['dvh' in x for x in self.loss_types]):
            oars = [f'contours/{n}' for n in user_inputs["DATA_PREPROCESSING"]["oars"]["name_and_voxel_value_pairs"].keys()]
            self.targets = self.targets + ["contours/body"] + ["contours/combined_targets"] + [x for x in oars if x != "contours/body"]
            
        elif any(['focused' in x for x in self.loss_types]):
            self.targets += ['contours/body']

        # Determine the number of channels
        self.no_channels = len(self.inputs)
        
        # Log information
        if self.__log_type == 'training':
            self.__logger.info(f'Selected model: {self.model_name}')
            self.__logger.info(f"Number of input channels detected: {self.no_channels}")
            self.__logger.info(f"Input features: {','.join(self.inputs)}")
            self.__logger.info(f"Target features: {','.join(self.targets)}")
            self.__logger.info(f"Fold used for training and validation: {self.k_fold}")
            self.__logger.info(f"Maximum number of epochs: {self.epochs}")
            self.__logger.info(f"Batch size: {self.batch_size}")
            self.__logger.info(f"Data augmentation types: {','.join(self.augmentation_types)}")
            self.__logger.info(f"Augmenting validation set: {self.augment_validation_dataset}")
            self.__logger.info(f"Training set repeats: {self.dataset_repeats['training-set']}")
            self.__logger.info(f"Validation set repeats: {self.dataset_repeats['validation-set']}")
            self.__logger.info(f"Initial learning rate: {self.initial_learning_rate}")
            self.__logger.info(f"Loss function types: {','.join(self.loss_types)}")
            self.__logger.info(f"Optimizer type: {self.optimizer['type']}")
            # if self.seed is None: self.seed = int(time.time())   
            # self.__logger.info(f"Random seed: {self.seed}")  
            self.__logger.info(f"Selected learning rate decay: {self.lr_decay['type']}\n")
            
        elif self.__log_type == 'evaluation':
            self.__logger.info(f"Data set to evaluate: {self.data_set_to_evaluate}")
            self.__logger.info(f"Epoch to evaluate: {self.eval_epoch}")
            self.__logger.info(f"Fold to evaluate: {self.eval_folds}")
            self.__logger.info(f"Patch dimensions: {self.eval_patch_dimensions}")
            self.__logger.info(f"Patch stride: {self.eval_patch_stride}")
            self.__logger.info(f"Data augmentation: {self.eval_augmentation}")
            self.__logger.info(f"Minimum dose: {self.min_dose}")
            self.__logger.info(f"Volume for maximum dose: {self.vol_for_max_dose}")
            self.__logger.info(f"Normalization method: {self.normalization_method}")