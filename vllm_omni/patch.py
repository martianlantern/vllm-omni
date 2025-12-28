import sys

from vllm.config import ModelConfig as _OriginalModelConfig
from vllm.inputs.data import TokensPrompt as _OriginalTokensPrompt
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding as _OriginalMRotaryEmbedding,
)
from vllm.v1.engine import EngineCoreOutput as _OriginalEngineCoreOutput
from vllm.v1.engine import EngineCoreOutputs as _OriginalEngineCoreOutputs
from vllm.v1.engine import EngineCoreRequest as _OriginalEngineCoreRequest
from vllm.v1.request import Request as _OriginalRequest

import vllm_omni.logger  # noqa: F401
from vllm_omni.config import OmniModelConfig
from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs, OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.layers.mrope import MRotaryEmbedding
from vllm_omni.request import OmniRequest

for module_name, module in sys.modules.items():
    # only do patch on module of vllm, pass others
    if "vllm" not in module_name:
        continue
    if hasattr(module, "EngineCoreOutput") and module.EngineCoreOutput == _OriginalEngineCoreOutput:
        module.EngineCoreOutput = OmniEngineCoreOutput
    if hasattr(module, "EngineCoreOutputs") and module.EngineCoreOutputs == _OriginalEngineCoreOutputs:
        module.EngineCoreOutputs = OmniEngineCoreOutputs
    if hasattr(module, "TokensPrompt") and module.TokensPrompt == _OriginalTokensPrompt:
        module.TokensPrompt = OmniTokensPrompt
    if hasattr(module, "MRotaryEmbedding") and module.MRotaryEmbedding == _OriginalMRotaryEmbedding:
        module.MRotaryEmbedding = MRotaryEmbedding
    if hasattr(module, "Request") and module.Request == _OriginalRequest:
        module.Request = OmniRequest
    if hasattr(module, "EngineCoreRequest") and module.EngineCoreRequest == _OriginalEngineCoreRequest:
        module.EngineCoreRequest = OmniEngineCoreRequest
    if hasattr(module, "ModelConfig") and module.ModelConfig == _OriginalModelConfig:
        module.ModelConfig = OmniModelConfig

# Patch model registry to include Omni models
# Patch model registry to include Omni models
try:
    import importlib

    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.registry import _VLLM_MODELS

    from vllm_omni.model_executor.models.registry import _OMNI_MODELS

    for model_arch, (mod_folder, mod_relname, cls_name) in _OMNI_MODELS.items():
        module_path = f"vllm_omni.model_executor.models.{mod_folder}.{mod_relname}"

        # Manually update _VLLM_MODELS to pass ModelConfig validation which might check this dict directly
        # Format expected: model_arch -> (module_path, class_name)
        _VLLM_MODELS[model_arch] = (module_path, cls_name)

        try:
            # Eagerly import the class since register_model expects a class or string (but string might fail lazy check)
            mod = importlib.import_module(module_path)
            model_cls = getattr(mod, cls_name)

            # Register the model class
            if hasattr(ModelRegistry, "register_model"):
                ModelRegistry.register_model(model_arch, model_cls)

        except Exception:
            # If import fails (e.g. missing deps), log or skip
            pass

except ImportError:
    pass

except ImportError:
    pass
