"""Reproduce keyword dispatch without requiring ComfyUI or model weights."""
import importlib
import inspect
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class Port:
    @staticmethod
    def Input(name, **kwargs):
        return types.SimpleNamespace(name=name, **kwargs)

    @staticmethod
    def Output(name, **kwargs):
        return types.SimpleNamespace(name=name, **kwargs)


class ContractTests(unittest.TestCase):
    def test_schema_keywords_reach_runtime_and_four_outputs_return(self):
        package = types.ModuleType("h3_contract")
        package.__path__ = [str(ROOT)]
        io = types.SimpleNamespace(ComfyNode=object, Schema=lambda **kw: types.SimpleNamespace(**kw),
                                   NodeOutput=lambda *values: values)
        for name in ("Model", "Clip", "Vae", "String", "Int", "Float", "Image", "Sampler", "Sigmas", "Audio", "Video", "Latent"):
            setattr(io, name, Port)
        comfy_api = types.ModuleType("comfy_api")
        latest = types.ModuleType("comfy_api.latest")
        latest.io = io
        seen = []
        fake_runtime = types.ModuleType("h3_contract.runtime")
        fake_runtime.H3Runtime = lambda *args, **kwargs: seen.append(kwargs) or "backend"
        with patch.dict(sys.modules, {"comfy_api": comfy_api, "comfy_api.latest": latest,
                                     "h3_contract": package, "h3_contract.runtime": fake_runtime}):
            module = importlib.import_module("h3_contract.nodes")
            cls = module.MiniMaxH3SimpleVideo
            schema = cls.define_schema()
            signature = inspect.signature(cls.execute)
            self.assertTrue(all(port.name in signature.parameters for port in schema.inputs))
            self.assertEqual([p.name for p in schema.outputs], ["video", "images", "audio", "latent"])
            audio, reference = object(), object()
            with patch.object(module, "generate", return_value=("v", "i", "a", "l")):
                result = cls.execute(model=None, clip=None, vae=None, audio_vae=None,
                                     prompt="test", seconds=10, width=256, height=256, seed=1,
                                     audio_input=audio, reference_image=reference)
                self.assertEqual(result, ("v", "i", "a", "l"))
                self.assertIs(seen[-1]["audio_input"], audio)
                self.assertIs(seen[-1]["reference_image"], reference)
                cls.execute(model=None, clip=None, vae=None, audio_vae=None,
                            prompt="test", seconds=5, width=256, height=256, seed=0)
                self.assertEqual(seen[-1], {"reference_image": None, "audio_input": None})


if __name__ == "__main__":
    unittest.main()
