# MiniMax Studio Long Video

One standalone ComfyUI node: enter one prompt, a total duration and a chunk size, then connect its VIDEO output to Save Video.

With the default **chunk_seconds = 5**:

- 5 seconds → one section.
- 12 seconds → 5 + 5 + 2 seconds.
- 30 seconds → six 5-second sections.
- One shared prompt embedding and one rolling video/audio context.
- Optional starting image. No timeline, project folders, API key or Easy Media dependency.

## Install

In ComfyUI Manager, use **Install via Git URL** with:

```text
https://github.com/comfyuistudio/MiniMax_Studio_Long_Video
```

Or clone the repository into `ComfyUI/custom_nodes`:

```sh
git clone https://github.com/comfyuistudio/MiniMax_Studio_Long_Video.git
```

Restart ComfyUI. Search for **MiniMax Studio Long Video** under **MiniMax Studio / Video**.
When upgrading from a ZIP installation, remove older copies from `custom_nodes` first to avoid duplicate node registrations.

Requires a current ComfyUI build with native MiniMax H3, `MiniMaxH3AddGuide` and `VideoFromList` support.
It uses ComfyUI's existing runtime dependencies and does not download models or install packages automatically.
The code uses the V3 custom-node API. This package is experimental: automated tests cover the orchestration,
but end-to-end GPU generation, continuity and lip-sync quality have not been validated by this project.

## Example workflows

- [Complete text-to-video workflow](example_workflows/MiniMax-H3-Simple-Complete-Workflow.json)
- [Reference image workflow](example_workflows/MiniMax-H3-Simple-Reference-Workflow.json)
- [Reference image and speech workflow](example_workflows/MiniMax-H3-Simple-LipSync-Workflow.json)
- [Minimal node and Save Video](example_workflows/simple-workflow.json)

Import a JSON workflow into ComfyUI, choose your installed model files and upload your own image/audio where applicable.

## Connect

Start with your working H3 text-to-video or image-to-video workflow and reuse its loaders:

| Input | Connect |
| --- | --- |
| model | Your loaded H3 **fl2va** model (or **ref2va** when reference_image is connected), after any existing model patches or LoRAs |
| clip | H3's Qwen text encoder |
| vae | H3 video VAE |
| audio_vae | H3 audio VAE |
| first_frame | Optional starting frame from Load Image |
| reference_image | Optional appearance reference from Load Image; requires H3 ref2va model |

Set **prompt**, **seconds**, **chunk_seconds**, **width**, **height**, and **seed**. Connect **video → Save Video** and queue the workflow. Audio is included automatically. A small importable `simple-workflow.json` contains this node and Save Video; connect your existing loaders to its four open inputs.

Default sampling is Euler, the model's simple schedule, 20 steps and CFG 1. Advanced controls expose steps and CFG. Optional SAMPLER and SIGMAS sockets let you reuse a known working sampling setup; a connected SIGMAS schedule overrides steps. Model patches, sigma shifts and turbo LoRAs remain upstream so the node does not overwrite your model settings. Use a full-denoise schedule ending in zero. It does not load or enable a turbo LoRA automatically.

## Audio input for lip-sync guidance

Connect **Load Audio → AUDIO** to **audio_input**. `MiniMax-H3-Simple-LipSync-Workflow.json` includes this connection plus the image reference. Upload your recording and image, select the H3 ref2va model, and describe a speaking person with a visible mouth in the prompt.

The node slices the recording on the global video timeline, including the context overlap and internal frame padding. Each slice is passed to native `MiniMaxH3AddGuide` as a time-aligned audio guide before sampling. The speech does not restart at every chunk, and generated context audio is omitted when supplied speech drives the clip. Native H3 handles resampling for its audio VAE.

VIDEO and AUDIO outputs preserve the original waveform at its input sample rate, trimmed to `seconds` or padded with silence when the recording is shorter. `seconds` still controls total duration; it is not inferred from the audio file. Supported input is one finite, nonempty mono or stereo waveform. Video encoding can compress audio. Without audio_input, generated audio behavior is unchanged.

This provides native audio guidance, not a dedicated lip-sync correction model. Mouth synchronization and identity consistency remain model-dependent and have not been visually validated on a GPU. LATENT contains the final sampled joint latent; its generated audio stream is not replaced with the original recording. Use the AUDIO output when you need the preserved speech.

## Reference image

Connect **Load Image → IMAGE** to **reference_image**. Use an H3 **ref2va** diffusion model and mention `<Picture 1>` in your prompt. The image is encoded once using native reference conditioning and retained across every chunk alongside the rolling motion context. One image is supported; image batches are rejected explicitly. The original `first_frame` input remains available as a separate starting-frame anchor. With no reference connected, the existing text/first-frame path is unchanged.

`MiniMax-H3-Simple-Reference-Workflow.json` includes the connection and reference-model loader. Upload your image before queuing. No example image or model downloads are bundled. This uses [ComfyUI's official reference-to-video interface](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py).

## Outputs

The node has four outputs, in this order:

| Socket | Contents |
| --- | --- |
| video | Complete video with audio; connect to Save Video |
| images | Complete trimmed frame batch, in timeline order at 24 fps |
| audio | Complete trimmed waveform and sample rate |
| latent | Final chunk's raw joint H3 video/audio latent, including overlap and grid padding |

IMAGE and AUDIO are retained before video compression. LATENT is not a stitched latent for the whole movie. It includes `h3_trim_frames`, `h3_output_frames` and `h3_fps` metadata describing the final chunk's visible portion. Standard VAE Decode nodes do not automatically apply this metadata: trim their decoded output explicitly if needed. Existing VIDEO connections keep socket index 0.

Returning all frames requires RAM proportional to total duration and resolution, even when IMAGE is not connected. Joining chunks temporarily adds another full frame buffer. At 1344×768, float32 RGB frames alone use roughly 2.8 GiB per 10 seconds. The final latent is moved to CPU; earlier chunk latents are released.

## Chunk size

**chunk_seconds** selects 1–14 whole seconds per output section; default: 5. For example, 20 total seconds with 8-second chunks produces 8 + 8 + 4 seconds. A chunk larger than the total duration produces one shortened section. Larger chunks require more memory. The maximum of 14 seconds leaves room for the 22-frame context within H3's 362-frame sampling window. Small chunks still generate at least 124 internal frames before trimming, so they may be less efficient. Existing API requests that omit chunk_seconds retain the 5-second default.

## Continuity and timing

The prompt is encoded once. Before each subsequent section, the last **22 delivered video frames** (about 0.92 seconds) and matching audio become native H3 guides at the start of the next generation. These overlap frames are removed before joining. The optional initial image's time-zero anchor is replaced by the rolling guide on subsequent sections.

H3's native grid is `17k + 5` frames at 24 fps. Internally the node generates at least 124 frames, adding room for context and grid padding; only the requested section is retained. Thus 5 seconds means exactly 120 **output** frames, not a promise that the internal sampling window is exactly five seconds. The final partial section is trimmed to the requested duration. Crucially, context comes from the last retained frame, not from discarded padding beyond the cut.

This is sequential generation with one rolling context, not a persistent model KV cache or unlimited memory of the whole film. It supports continuity but cannot guarantee seamless motion, identity, dialogue or audio across every boundary. Describe a continuous action and camera movement in the prompt. Timed actions in the same prompt can repeat across sections.

Each section is encoded to H.264 immediately, then combined using ComfyUI's native video API. Trimmed frames and audio are also retained on CPU for the IMAGE and AUDIO outputs. Compression is lossy. Duration is limited to 300 whole seconds; generation time grows with the number of sections. Cancellation is checked between sections and handled by ComfyUI during sampling. Intermediate sections are not persisted as recovery checkpoints.

## Validation

Run `python -m unittest discover -s tests -v` from this folder. Tests cover timing, partial sections, seeds, context chaining, trimming, cancellation and the four output values with a mocked backend. No model weights or GPU are required for these checks.

Real H3 generation and visual seam quality have **not** been validated on this machine; ComfyUI and PyTorch are not available in its command-line Python environment. Test a 10-second clip in your working H3 installation first.

## References

ComfyUI Easy Media was inspected to understand its motion-context behavior. This smaller package delegates to native ComfyUI nodes and does not import or copy the Easy Media implementation.

- [Native MiniMax H3 implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py)
- [ComfyUI video implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_video.py)
- [Official H3 guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native)
