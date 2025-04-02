from abc import ABC, abstractmethod
import re
from typing import Optional, List, Tuple, Any, Dict
from copy import deepcopy
from transformers import AutoTokenizer
import torch
from PIL import Image
import numpy as np
from dataclasses import dataclass, field


@dataclass
class BaseConifg():
    init_config:Any # for interface initialization
    reset_config:Any # for interface reset