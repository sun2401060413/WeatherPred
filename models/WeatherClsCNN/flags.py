'''
仿absl.flags功能，限制更少
'''
import absl.flags as app_flags
class FLAGS():
    def __init__(self, info_dict = {}):
        self.info_dict = info_dict
    def define(self, varname, defaule_value):
        if varname in self.info_dict.keys():
            return self.info_dict[varname]
        else:
            return defaule_value

# interface class
class DEFAULT_FLAGS():                                  # 还存在部分逻辑问题
    def __init__(self, FLAGS = app_flags.FLAGS):
        self.FLAGS = FLAGS

        self.mode = None
        self.filepath = None
        self.savepath = None
        self.weightpath = None
        self.modelname = None
        self.input_mode = None
        self.extension = None
        self.training_ratio = None
        self.validation_ratio = None
        self.size = None
        self.channels = None
        self.batch_size = None
        self.epoches = None
        self.lr = None
        self.op = None
        self.loss = None

        self.info_dict = dict()

    def get_default_dict(self):
        if self.FLAGS.filepath != None:
           self.info_dict["filepath"] = self.FLAGS.filepath
        if self.FLAGS.savepath != None:
            self.info_dict["savepath"] = self.FLAGS.savepath
        if self.FLAGS.weightpath != None:
            self.info_dict["weightpath"] = self.FLAGS.weightpath
        if self.FLAGS.extension != None:
           self.info_dict["extension"] = self.FLAGS.extension
        else:
            self.info_dict["extension"] = [".jpg", ".png"]
        self.info_dict["mode"] = self.FLAGS.mode
        self.info_dict["modelname"] = self.FLAGS.modelname
        self.info_dict["input_mode"] = self.FLAGS.input_mode
        self.info_dict["training_ratio"] = self.FLAGS.training_ratio
        self.info_dict["validation_ratio"] = self.FLAGS.validation_ratio
        self.info_dict["size"] = self.FLAGS.size
        self.info_dict["channels"] = self.FLAGS.channels
        self.info_dict["batch_size"] = self.FLAGS.batch_size
        self.info_dict["epoches"] = self.FLAGS.epoches
        if self.FLAGS.lr != None:
            self.info_dict["lr"] = self.FLAGS.lr
        if self.FLAGS.op != None:
            self.info_dict["op"] = self.FLAGS.op
        if self.FLAGS.loss != None:
            self.info_dict["loss"] = self.FLAGS.loss
        # self.info_dict = info_dict
        return self.info_dict

    def set_value(self):
        return self.info_dict