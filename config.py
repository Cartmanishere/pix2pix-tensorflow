import os


class DefaultConfig(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, parsed_args=None):
        super().__init__()

        ### Constants ###
        self.aspect_ratio = 1.0
        self.batch_size = 1
        self.beta1 = 0.5
        self.flip = False
        self.gan_weight = 1.0
        self.l1_weight = 100.0
        self.lab_colorization = False
        self.lr = 0.0002
        self.max_epochs = 500
        self.max_steps = None
        self.ndf = 64
        self.ngf = 32
        self.output_filetype = 'png'
        self.scale_size = 256
        self.seed = 1040378113
        self.separable_conv = False
        self.mode = 'train'
        self.trace_freq = 0
        self.which_direction = 'BtoA'
        self.CROP_SIZE = 256
        self.EPS = 1e-12
        
        ### Frequencies ###
        self.progress_freq = 10
        self.save_freq = 10 
        self.display_freq = 10
        self.summary_freq = 10
        
        ### Directories ###
        self.output_dir = 'model_output/'
        self.test_dir = 'test_data'
        self.train_dir = 'train_data/'
        self.checkpoint = None
        
        if parsed_args is not None:
            self.parse_args(parsed_args)
        
        self.assure_path(self.output_dir)
        self.assure_path(self.test_dir)
        self.assure_path(self.train_dir)
        self.assure_path(self.checkpoint)

    def parse_args(self, parsed_args):
        args = vars(parsed_args)
        for k, v in args.items():
            if v is not None:
                self[k] = v

    def assure_path(self, path):
        if path is None:
            return
        
        # check relative or absolute
        if path[0] != '/':
            path = os.path.join(os.getcwd(), path)

        if not os.path.isdir(path):
            os.mkdir(path)
