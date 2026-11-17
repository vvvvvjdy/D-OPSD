# Practical Recommendations

These are our recommended starting points for D-OPSD adaptation. Adjust them to the model, task, and observed fitting behavior. Existing example settings and candidate models are identified separately below.

## Recommended defaults

| Choice | Recommendation |
| --- | --- |
| Initial model | Start with a step-distilled checkpoint |
| Teacher context | For a model with native editing/reference support, start with that capability; explore other useful context designs as appropriate |
| Teacher update | **EMA decay 0.9999** by default; also consider a fixed teacher |
| Loss | **MSE between teacher and student predicted clean latents (`x_0`)** at the same student state and time |
| Sampling | Start with the model's native default step count, actual time points/values, sampler, and guidance |
| Guidance | Step-distilled models generally use no extra CFG, commonly `guidance_scale=1`; follow the native pipeline's convention |
| Small-domain LoRA | Around **10 high-quality examples** can be a useful starting point for a focused subject, style, or narrow task |
| Full fine-tuning learning rate | Start **5–10 times lower than the corresponding LoRA learning rate**, then adjust according to fitting behavior |
| Final inference | Load student weights/LoRA into the original pipeline, retaining its normal inputs and sampling configuration |

The small-data recommendation concerns focused adaptation. The number and diversity of examples should reflect the variation the user wants the model to handle. Use representative examples and prompts, and inspect both learning of the target behavior and responses to new requests.

## Teacher EMA and fixed teachers

Use **0.9999** as the default EMA decay. This keeps the teacher changing slowly while allowing it to track the student. A fixed teacher is another useful option, especially when its initial context-conditioned behavior already provides good supervision.

For a fixed teacher, keep the initialized teacher parameters unchanged and skip EMA updates. Disabling teacher gradients alone does not make it fixed if an EMA update is still applied.

In the current public examples, launch scripts explicitly set `--ema-decay 0.9999`, while the argument parser defaults to `0.9`. **Set the recommended value explicitly in a new configuration.** Refer to the [paper's teacher ablation](https://arxiv.org/html/2605.05204v1#S3.SS3) for the original comparison.

## Learning rates and LoRA settings

Current public launch scripts provide these concrete examples:

| Example | LoRA LR | Rank / alpha | Teacher EMA |
| --- | --- | --- | --- |
| [FLUX.2-klein-4B customization](https://github.com/vvvvvjdy/D-OPSD/blob/main/flux2-klein_self-distill-edit/scripts/train_lora_4b.sh) | `2e-5` | `64 / 128` | `0.9999` |
| [FLUX.2-klein-4B editing](https://github.com/vvvvvjdy/D-OPSD/blob/main/flux2-klein-edit-self-distill-gt-ref/scripts/train_lora_4b.sh) | `2e-5` | `64 / 128` | `0.9999` |
| [Z-Image-Turbo customization](https://github.com/vvvvvjdy/D-OPSD/blob/main/z-image-turbo_self-distill-vlm/scripts/train_lora.sh) | `1e-4` | `64 / 128` | `0.9999` |

These values describe the supplied examples. Select LoRA modules according to the new model's architecture, and treat rank, alpha, and LR as settings to adapt.

For full fine-tuning, use roughly one fifth to one tenth of the corresponding LoRA LR initially. For example, a LoRA starting LR of `2e-5` suggests an initial full fine-tuning range of `2e-6` to `4e-6`. This illustrates the recommendation; adjust further according to fitting behavior and sample quality.

Track progress with fixed inputs and seeds as well as loss. If the teacher context is unhelpful, revisit it before trying to solve the problem through LR or training duration alone.

## Sampling steps and time values

For a new model, begin with the native default number of sampling steps. Prefer keeping the **actual sampling time points and values identical** to the native inference schedule, including shifts or other schedule transformations.

Small variations around the default step count can be explored. Record the resulting timestep/sigma array and validate generation quality. Check the actual model evaluations and scheduler behavior rather than relying only on a configuration field named `num_inference_steps`.

For the zero-shot teacher comparison, hold the sampling settings fixed and change only the conditioning. The teacher does not need extra sampling steps or stronger CFG to serve this role.

### Why the Z-Image example trains with four steps

Our internal Z-Image distillation setup used **four sampling steps**. We therefore used the corresponding four-step rollout configuration in the subsequent D-OPSD training, and the public script carries over that setting. The reason for the four-step choice is to follow the sampling setup used during our internal distillation. Four-step training is not a general requirement of D-OPSD or a default to copy to other checkpoints.

For a new adaptation, use the target step-distilled checkpoint's native default step count and actual sampling time values first. The internal setup behind our Z-Image example does not establish that an arbitrary new model can use the same four-step schedule.

The [public Z-Image inference example](https://github.com/vvvvvjdy/D-OPSD/blob/main/z-image-turbo_self-distill-vlm/README.md) retains the model's eight effective denoising steps. Its demonstrated Diffusers call uses `num_inference_steps=9`, which the example notes corresponds to eight DiT forwards, and `guidance_scale=0.0`. Preserve the behavior of the pipeline version in use and verify its actual schedule. This also illustrates why API argument values should be interpreted through the model's native implementation.

## Models to consider

The repository provides Z-Image-Turbo and FLUX.2-klein examples. The models below are **candidates for new adaptations**, not a statement that this repository already provides or has validated D-OPSD implementations for them.

Official model information was checked on **2026-11-17**. Recheck the exact checkpoint and pipeline when implementing a port.

| Model | Step-distilled checkpoint | Official default sampling | Porting notes |
| --- | --- | --- | --- |
| FLUX 3 Image Klein | `black-forest-labs/flux-3-image-klein` | 16 steps, no extra CFG | The `-base` variant is undistilled. Preserve the native conditioning and prompt-upsample workflow. [Official documentation](https://docs.bfl.ai/flux_3/flux3_image_klein) |
| MiniMax-H3-Image | `MiniMaxAI/MiniMax-H3-Image` | 6 steps / 6 NFE, no extra CFG | Inspect its complete sampling implementation, including step-specific prediction behavior, when extracting a trainable path. [Official model card](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) |
| Qwen-Image-2.1-Turbo | `Qwen/Qwen-Image-2.1-Turbo` | 8 steps, CFG=1 | The checkpoint includes its recommended schedule. Its model card says changing `num_inference_steps` alone does not override that schedule. [Official model card](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) |

The exact context design remains task-dependent. Start by inspecting zero-shot teacher results with the original inference settings, then adapt the D-OPSD training loop. We welcome suggestions and reports for these and other models in [GitHub Issues](https://github.com/vvvvvjdy/D-OPSD/issues).
