#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .cnn import Cnn, CnnConfig
from .common import (
    EnsembleModelConfig,
    Model,
    ModelConfig,
    SequenceModel,
    SequenceModelConfig,
)
from .deepsets import Deepsets, DeepsetsConfig
from .gnn import Gnn, GnnConfig
from .gru import Gru, GruConfig
from .lstm import Lstm, LstmConfig
from .mlp import Mlp, MlpConfig

from .pimore import PIMoRE, PIMoREConfig
from .maac import MAAC, MAACConfig
from .mapoca import MAPOCA, MAPOCAConfig

classes = [
    "Mlp",
    "MlpConfig",
    "Gnn",
    "GnnConfig",
    "Cnn",
    "CnnConfig",
    "Deepsets",
    "DeepsetsConfig",
    "Gru",
    "GruConfig",
    "Lstm",
    "LstmConfig",
    "PIMoRE",
    "PIMoREConfig",
    "MAAC",
    "MAACConfig",
    "MAPOCA",
    "MAPOCAConfig",
]

model_config_registry = {
    "mlp": MlpConfig,
    "gnn": GnnConfig,
    "cnn": CnnConfig,
    "deepsets": DeepsetsConfig,
    "gru": GruConfig,
    "lstm": LstmConfig,
    "pimore": PIMoREConfig,
    "maac": MAACConfig,
    "mapoca": MAPOCAConfig,
}
