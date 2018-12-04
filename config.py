import os
import absl.flags as flags
import torch
import torch.backends.cudnn as cudnn

home = os.path.expanduser(".")

flags.DEFINE_string("mode","train","mode")

flags.DEFINE_string("data_pickle","./data/activity-net/data.pickle","data pickle")
flags.DEFINE_string("p_train_data","./data/activity-net/p_train_data.json","processed train_data")
flags.DEFINE_string("p_val_data","./data/activity-net/p_val_data.json","processed tval_data")
flags.DEFINE_string("p_test_data","./data/activity-net/p_test_data.json","processed ttest_data")
flags.DEFINE_string("train_data","./data/activity-net/train_data.json","train_data")
flags.DEFINE_string("val_data","./data/activity-net/val_data.json","val_data")
flags.DEFINE_string("test_data","./data/activity-net/test_data.json","test_data")
flags.DEFINE_string("feature_path","/home1/zhangzhu/data/activity-c3d/","feature_path")
flags.DEFINE_string("cache_dir","./results/baseline/","cache_dir")

flags.DEFINE_string("word2vec","/home1/xiaozhenxin/word2vec/word2vec.bin","word2vec path")
flags.DEFINE_integer("input_ques_dim", 300, "Embedding dimension for Glove")
flags.DEFINE_integer("input_video_dim", 500, "Embedding dimension for Glove")
flags.DEFINE_integer("max_frames", 200, "Limit length for paragraph")
flags.DEFINE_integer("max_words", 20, "Limit length for question")
flags.DEFINE_integer("evaluate_interval", 3, "evaluate_interval")
flags.DEFINE_integer("max_epoches", 1000, "max_epoches")
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("num_steps", 60000, "Number of steps")
flags.DEFINE_integer("display_batch_interval", 10, "display_batch_interval")
flags.DEFINE_integer("checkpoint", 1000, "checkpoint to save and evaluate the model")
flags.DEFINE_integer("period", 100, "period to save batch loss")
flags.DEFINE_integer("val_num_batches", 150, "Number of batches to evaluate the model")
flags.DEFINE_integer("test_num_batches", 150, "Number of batches to evaluate the model")
flags.DEFINE_float("dropout", 0.1, "Dropout prob across the layers")
flags.DEFINE_float("dropout_char", 0.05, "Dropout prob across the layers")
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
flags.DEFINE_integer("lr_warm_up_num", 1000, "Number of warm-up steps of learning rate")
flags.DEFINE_float("ema_decay", 0.9999, "Exponential moving average decay")
flags.DEFINE_float("beta1", 0.8, "Beta 1")
flags.DEFINE_float("beta2", 0.999, "Beta 2")
# flags.DEFINE_float("l2_norm", 3e-7, "L2 norm scale")
flags.DEFINE_integer("early_stopping", 10, "Checkpoints for early stop")
flags.DEFINE_integer("connector_dim", 96, "Dimension of connectors of each layer")
flags.DEFINE_integer("num_heads", 2, "Number of heads in multi-head attention")

config = flags.FLAGS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# cudnn.enabled = False