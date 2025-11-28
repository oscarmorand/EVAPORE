import os
import json

class ConfigManager:
    def __init__(self, processed_dir):
        self.processed_dir = os.path.join(processed_dir, "dynamic_dataset")
        os.makedirs(self.processed_dir, exist_ok=True)
        self.cfgs_path = os.path.join(self.processed_dir, "configs.json")
        if not os.path.exists(self.cfgs_path):
            with open(self.cfgs_path, 'w') as f:
                json.dump({}, f)


    @classmethod
    def config_equals(cls, cfg1: dict, cfg2: dict) -> bool:
        if not isinstance(cfg1, dict):
            if not isinstance(cfg2, dict):
                return cfg1 == cfg2
            return False
        if not isinstance(cfg2, dict):
            return False
        for key in cfg1.keys():
            if key not in cfg2:
                return False
            if not cls.config_equals(cfg1[key], cfg2[key]):
                return False
        for key in cfg2.keys():
            if key not in cfg1:
                return False
        return True


    def get_existing_configs(self):
        if os.path.exists(self.cfgs_path):
            with open(self.cfgs_path, 'r') as f:
                json_data = json.load(f)
            return json_data
        return {}
    

    def already_exists(self, cfg: dict) -> bool:
        json_data = self.get_existing_configs()
        for _, val in json_data.items():
            cfg_to_compare = val.get('config')
            if self.config_equals(cfg, cfg_to_compare):
                path = val.get('path')
                return True, path
        return False, None


    def add_config(self, cfg: dict) -> None:
        json_data = self.get_existing_configs()

        existing_ids = [int(k) for k in json_data.keys()]
        if existing_ids and len(existing_ids) > 0:
            new_id = str(max(existing_ids) + 1)
        else:
            new_id = "0"
        new_dir = os.path.join(self.processed_dir, f"dynamic_dataset_{new_id}")

        if new_id in json_data:
            raise ValueError(f"Config with id {new_id} already exists.")
        
        json_data[new_id] = {
            "config": cfg,
            "path": new_dir
        }
        with open(self.cfgs_path, 'w') as f:
            json.dump(json_data, f, indent=4)

        return new_dir
    

    def save_single_config(self, cfg: dict, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        cfg_path = os.path.join(path, "config.json")
        with open(cfg_path, 'w') as f:
            json.dump(cfg, f, indent=4)