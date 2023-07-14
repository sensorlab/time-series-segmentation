def Config_declaration(name, 
                       model = "first",
                       seed = 75,
                       learning_rate = 0.0005,
                       batch_size =  4, #32,
                       range_epoch = 3000,
                       save_file = " test_test",
                       name_of_save = "test_u-time",
                       patience = 400,
                       loss = "CE", #"BCE", "CE"
                       train = 0.8,
                       val = 0.2, #of train
                       test = 0.2
                      ):
    if name == "TSSB":
        Config = {   
            "name":"TSSB",
            "dataset_path":"datasets/TSSB/",

            "graph":{
                "custom" : False,
                "masking" : True,
                "one_graph" :True,
                "type_number" : 20,
                "classif":"graph", #"node","graph"
                "type":"VG", #"MTF", "VG", "Dual_VG"
                "MTF":{
                    "num_bins":"auto" #you can imput an intiger or "auto" in which case it will chose the length of X as number of bins
                },
                "VG":{
                    "edge_type": "natural", #"natural", "horizontal"
                    "distance": 'distance', #'slope', 'abs_slope','distance','h_distance','v_distance','abs_v_distance',
                    "edge_dir": "directed" #"undirected", "directed"
                },
                "stride_param":{
                    "stride_type": "notflexible_with_remainder", #"flexible", "notflexible", "notflexible_with_remainder"
                }
            },

            "main":{
                "SEED": seed,
                "learning_rate": learning_rate,
                "batch_size": batch_size, 
                "range_epoch": range_epoch,
                "save_file": save_file,
                "name_of_save": name_of_save,
                "patience": patience,
                "loss": loss
                
            },
            "train/val/test":{
                "train":train,
                "val":val, #of train
                "test":test
            },
            "keep_last": True
        }
    elif name == "U-time":
        Config = {   
            "name":"U-time",
            "U-time":{
                "people_batch" : 5,
                "sample" : 1
            },

            "graph":{
                "custom" : False,
                "masking" : False,
                "one_graph" :True,
                "classif":"graph", #"node","graph"
                "type":"VG", #"MTF", "VG", "Dual_VG"
                "MTF":{
                    "num_bins":"auto" #you can imput an intiger or "auto" in which case it will chose the length of X as number of bins
                },
                "VG":{
                    "edge_type": "natural", #"natural", "horizontal"
                    "distance": 'distance', #'slope', 'abs_slope','distance','h_distance','v_distance','abs_v_distance',
                    "edge_dir": "directed" #"undirected", "directed"
                },
                "stride_param":{
                    "stride_type": "notflexible_with_remainder", #"flexible", "notflexible", "notflexible_with_remainder"
                }
            },

            "main":{
                "SEED": seed,
                "learning_rate": learning_rate,
                "batch_size": batch_size, 
                "range_epoch": range_epoch,
                "save_file": save_file,
                "name_of_save": name_of_save,
                "patience": patience,
                "loss": loss
            },
            "train/val/test":{
                "train":train,
                "val":val, #of train
                "test":test
            },
            "keep_last": True
        }

    elif name == "Rutgers":
        Config = {   
            "name":"Rutgers",
            "dataset_paths":{
                    "path_main": "datasets/dataset_uncut.csv", #"dataset_uncut.csv", "dataset_cut.csv", "dataset_rss.npz"
                    "path_properties": "datasets/dataset_properties.npz",
                    "path_mask": "datasets/dataset_mask.npz",
                    "len_type": "un/cut" # un/cut, random
            },

            "graph":{
                "custom" : False,
                "masking" : False,
                "one_graph" :True,
                "classif":"graph", #"node","graph"
                "type":"VG", #"MTF", "VG", "Dual_VG"
                "MTF":{
                    "num_bins":"auto" #you can imput an intiger or "auto" in which case it will chose the length of X as number of bins
                },
                "VG":{
                    "edge_type": "natural", #"natural", "horizontal"
                    "distance": 'distance', #'slope', 'abs_slope','distance','h_distance','v_distance','abs_v_distance',
                    "edge_dir": "directed" #"undirected", "directed"
                },
                "stride_param":{
                    "stride_type": "notflexible_with_remainder", #"flexible", "notflexible", "notflexible_with_remainder"
                }
            },

            "main":{
                "SEED": seed,
                "learning_rate": learning_rate,
                "batch_size": batch_size, 
                "range_epoch": range_epoch,
                "save_file": save_file,
                "name_of_save": name_of_save,
                "patience": patience,
                "loss": loss
            },
            "train/val/test":{
                "train":train,
                "val":val, #of train
                "test":test
            },
            "keep_last": True
        }
    return Config