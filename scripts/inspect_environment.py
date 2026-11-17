"""Show package isolation separately from the underlying Python installation."""
import importlib.util
import json
import os
from pathlib import Path
import site
import sys

if __name__=="__main__":
    cfg=Path(sys.prefix)/"pyvenv.cfg"
    packages={}
    for name in ["torch","numpy","sklearn","funasr","opensmile","fastapi"]:
        spec=importlib.util.find_spec(name)
        packages[name]=spec.origin if spec else None
    print(json.dumps({"executable":sys.executable,"prefix":sys.prefix,"base_prefix":sys.base_prefix,"stdlib":os.__file__,"user_site_enabled":site.ENABLE_USER_SITE,"pyvenv_cfg":cfg.read_text() if cfg.exists() else None,"packages":packages},ensure_ascii=False,indent=2))
