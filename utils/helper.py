import dataclasses, json
from omegaconf import MISSING
from gflownet.config import Config
from gflownet.algo.config import TBVariant


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def custom_asdict_factory(data):
    def convert_value(obj):
        if isinstance(obj, TBVariant):
            return obj.value
        return obj

    return dict((k, convert_value(v)) for k, v in data if v != MISSING)


def strip_missing(cfg: Config) -> dict:
    """
    Recursively remove all fields in cfg that value MISSING
    """
    return dataclasses.asdict(cfg, dict_factory=custom_asdict_factory)
