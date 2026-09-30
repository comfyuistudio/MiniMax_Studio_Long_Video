from comfy_api.latest import ComfyExtension
from .nodes import MiniMaxH3SimpleVideo

__version__ = "0.2.2"
print(f"[MiniMax Studio Long Video v{__version__}] Loaded from {__file__}")


class Extension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3SimpleVideo]


async def comfy_entrypoint():
    return Extension()
