import os
import json
import random
import re
import shutil
import ipaddress
from urllib.parse import urlparse

import lazyllm
from lazyllm import launchers, LazyLLMCMD, ArgsDict, LOG
from .base import LazyLLMDeployBase, verify_func_factory
from .utils import get_log_path, make_log_dir

lazyllm.config.add('mindie_home', str, '', 'MINDIE_HOME')

verify_fastapi_func = verify_func_factory(error_message='Service Startup Failed',
                                          running_message='Daemon start success')
class Mindie(LazyLLMDeployBase):
    keys_name_handle = {
        'inputs': 'prompt',
    }
    default_headers = {'Content-Type': 'application/json'}
    message_format = {
        'prompt': 'Who are you ?',
        'stream': False,
        'max_tokens': 4096,
        'presence_penalty': 1.03,
        'frequency_penalty': 1.0,
        'temperature': 0.5,
        'top_p': 0.95
    }
    auto_map = {
        'port': int,
        'tp': ('world_size', int),
        'max_input_token_len': ('maxInputTokenLen', int),
        'max_prefill_tokens': ('maxPrefillTokens', int),
        'max_seq_len': ('maxSeqLen', int)
    }

    def __init__(self, trust_remote_code=True, launcher=launchers.remote(), log_path=None, **kw):  # noqa B008
        super().__init__(launcher=launcher)
        assert lazyllm.config['mindie_home'], 'Ensure you have installed MindIE and \
                                  "export LAZYLLM_MINDIE_HOME=/path/to/mindie/latest"'
        self.mindie_home = lazyllm.config['mindie_home']
        self.mindie_config_path = os.path.join(self.mindie_home, 'mindie-service/conf/config.json')
        self.backup_path = self.mindie_config_path + '.backup'
        self.custom_config = kw.pop('config', None)
        self.kw = ArgsDict({
            'npuDeviceIds': [[0]],
            'worldSize': 1,
            'port': 'auto',
            'host': '0.0.0.0',
            'maxSeqLen': 64000,
            'maxInputTokenLen': 4096,
            'maxPrefillTokens': 8192,
        })
        self.trust_remote_code = trust_remote_code
        self.kw.check_and_update(kw)
        self.kw['npuDeviceIds'] = [[i for i in range(self.kw.get('worldSize', 1))]]
        self.random_port = False if 'port' in kw and kw['port'] and kw['port'] != 'auto' else True
        self.temp_folder = make_log_dir(log_path, 'mindie') if log_path else None

        if self.custom_config:
            self.config_dict = (ArgsDict(self.load_config(self.custom_config))
                                if isinstance(self.custom_config, str) else ArgsDict(self.custom_config))
            self.kw['host'] = self.config_dict["ServerConfig"]["ipAddress"]
            self.kw['port'] = self.config_dict["ServerConfig"]["port"]
        else:
            default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mindie', 'config.json')
            self.config_dict = ArgsDict(self.load_config(default_config_path))

    def __del__(self):
        if hasattr(self, 'backup_path') and os.path.isfile(self.backup_path):
            shutil.copy2(self.backup_path, self.mindie_config_path)

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config_dict = json.load(file)
        return config_dict

    def save_config(self):
        if os.path.isfile(self.mindie_config_path):
            shutil.copy2(self.mindie_config_path, self.backup_path)

        with open(self.mindie_config_path, 'w') as file:
            json.dump(self.config_dict, file)

    def generate_remote_command(self):
        config_json = json.dumps(self.config_dict)
        # Escape double quotes to avoid shell issues
        escaped_config = config_json.replace('"', '\\"')
        cmd = (
            f'[ -f {self.mindie_config_path} ] && cp {self.mindie_config_path} {self.backup_path} ; '
            f'echo "{escaped_config}" > {self.mindie_config_path} ; '
        )
        if self.temp_folder: cmd += f"mkdir -p {self.temp_folder} ; "
        return cmd

    def update_config(self):
        backend_config = self.config_dict["BackendConfig"]
        backend_config["npuDeviceIds"] = self.kw["npuDeviceIds"]
        model_config = {
            "modelName": self.finetuned_model.split('/')[-1],
            "modelWeightPath": self.finetuned_model,
            "worldSize": self.kw["worldSize"],
            "trust_remote_code": self.trust_remote_code
        }
        backend_config["ModelDeployConfig"]["ModelConfig"][0].update(model_config)
        backend_config["ModelDeployConfig"]["maxSeqLen"] = self.kw["maxSeqLen"]
        backend_config["ModelDeployConfig"]["maxInputTokenLen"] = self.kw["maxInputTokenLen"]
        backend_config["ScheduleConfig"]["maxPrefillTokens"] = self.kw["maxPrefillTokens"]
        self.config_dict["BackendConfig"] = backend_config
        if self.kw["host"] != '0.0.0.0':
            self.config_dict["ServerConfig"]["ipAddress"] = self.kw["host"]
        self.config_dict["ServerConfig"]["port"] = self.kw["port"]

    def cmd(self, finetuned_model=None, base_model=None, master_ip=None):
        if self.custom_config is None:
            self.finetuned_model = finetuned_model
            if finetuned_model or base_model:
                if not os.path.exists(finetuned_model) or \
                    not any(filename.endswith('.bin') or filename.endswith('.safetensors')
                            for filename in os.listdir(finetuned_model)):
                    if not finetuned_model:
                        LOG.warning(f"Note! That finetuned_model({finetuned_model}) is an invalid path, "
                                    f"base_model({base_model}) will be used")
                    self.finetuned_model = base_model

            if self.random_port:
                self.kw['port'] = random.randint(30000, 40000)

            self.update_config()

        # self.save_config()

        def impl():
            config_cmd = self.generate_remote_command()
            cmd = config_cmd + f'{os.path.join(self.mindie_home, "mindie-service/bin/mindieservice_daemon")}'
            cmd += f" --port={self.kw['port']}"
            if self.temp_folder: cmd += f' 2>&1 | tee {get_log_path(self.temp_folder)}'
            return cmd

        return LazyLLMCMD(cmd=impl, return_value=self.geturl, checkf=verify_fastapi_func)

    def geturl(self, job=None):
        if job is None:
            job = self.job
        if lazyllm.config['mode'] == lazyllm.Mode.Display:
            return f'http://{job.get_jobip()}:{self.kw["port"]}/generate'
        else:
            ip_or_url = job.get_jobip()
            try:
                ipaddress.ip_address(ip_or_url)
                LOG.info(f"MindIE Server running on http://{ip_or_url}:{self.kw['port']}")
                return f'http://{ip_or_url}:{self.kw["port"]}/generate'
            except ValueError:
                pass

            parsed = urlparse(ip_or_url)
            if parsed.scheme in ("http", "https") and parsed.netloc:
                return ip_or_url + "/generate"

            raise ValueError(f"Not a valid IP or URL: {ip_or_url}")

    @staticmethod
    def extract_result(x, inputs):
        try:
            return json.loads(x)['text'][0]
        except Exception:
            try:
                json_strs = re.findall(r'\{.*?\}', x)
                texts = []
                for item in json_strs:
                    obj = json.loads(item)
                    texts.extend(obj.get("text", []))
                return "".join(texts)
            except Exception as e:
                LOG.warning(f'JSONDecodeError on load {x!r}')
                raise e

    @staticmethod
    def stream_url_suffix():
        return ''
