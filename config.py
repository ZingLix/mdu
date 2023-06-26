import yaml

with open("config/config.yml") as f:
    config = yaml.safe_load(f)

# bert配置
base_path = config["model"]["config"]
config_path = base_path + "bert_config.json"
checkpoint_path = base_path + "bert_model.ckpt"
dict_path = base_path + "vocab.txt"
maxlen = 512

data_path = "./data/MultiWOZ_2.2/test"
