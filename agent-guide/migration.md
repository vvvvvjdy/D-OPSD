# Migrating D-OPSD to a New Model or Dataset

Read the [method overview](README.md) first. Follow this workflow using the target model's native inference implementation and the closest existing D-OPSD example.

The goal is a working adaptation that preserves the core training procedure and produces student weights or LoRA usable by the original inference pipeline.

## 1. Establish the task and the native inference baseline

Identify the target checkpoint, desired improvement, available data, LoRA or full fine-tuning mode, and hardware. Use information already supplied by the user; ask about missing details that affect the implementation.

Start from a step-distilled checkpoint and run its official inference example on a few representative inputs. Keep the results as the baseline. Inspect the actual pipeline code and configuration, and record the following:

| Item | What to establish |
| --- | --- |
| Checkpoint and dependencies | Exact model variant and library revision; distinguish distilled and base variants |
| Conditioning | Text/image encoders, prompt template, hidden-state selection, masks, reference ordering, and any native prompt expansion |
| Latent representation | VAE normalization, latent shape, packing/unpacking, position IDs, and generated versus conditioning tokens |
| Prediction | Velocity, noise, clean latent, or another parameterization; any step-specific output heads |
| Sampling | Actual timestep/sigma array, time direction and scaling, solver, terminal step, and resolution-dependent schedule shifts |
| Guidance | The checkpoint's normal guidance behavior and the pipeline argument that expresses it |
| Trainable modules | Appropriate LoRA target modules or the parameters to update during full fine-tuning |

**Use the native default sampling settings first, including the actual time points and their values.** Matching `num_inference_steps` alone is insufficient if the scheduler produces a different array. Small changes around the native step count can be explored later, with their schedules recorded and their results checked.

Step-distilled models generally run without additional classifier-free guidance, commonly represented by `guidance_scale=1`. Preserve the target pipeline's convention: some pipelines express the no-CFG setting differently. Keep teacher and student guidance behavior consistent with the native model.

The Z-Image public example's four-step training schedule has a specific origin in our internal distillation setup. Read [the explanation](recipes.md#why-the-z-image-example-trains-with-four-steps) before reusing it.

## 2. Map the dataset to student and teacher conditions

Write down what each sample means and what will be available at inference. A small table is sufficient:

| Field | Meaning | Used by student? | Used by teacher? |
| --- | --- | --- | --- |
| Prompt/instruction | Normal request | Yes | Yes, with any teacher-specific instruction |
| Source/reference images | Inputs required by the deployed task, if any | According to the task | According to the task |
| Target image or other privileged information | Desired result or information to learn from | No, when unavailable at inference | Yes |

For text-to-image customization, the training reference image can be teacher-only. For editing, the source image remains a normal student input, while the target image can provide extra teacher context.

Adapt the loader to the user's existing schema. Preserve the meaning and ordering of images, check paths and nonempty prompts, and inspect a few decoded samples after preprocessing. For paired editing data, check that resizing/cropping preserves the intended relationship between source and target.

The existing Klein editing loader accepts lists of image paths and treats the **last image as the target**, with preceding images as source/reference inputs. This is an example schema, not a requirement for new loaders. See its [dataset implementation](https://github.com/vvvvvjdy/D-OPSD/blob/main/flux2-klein-edit-self-distill-gt-ref/dataset.py).

Keep validation relevant to the intended domain. A small-domain customization task should include new prompts or editing requests from that domain. Training examples can be useful for progress visualization; separately identify the examples used to assess transfer to new requests.

## 3. Check zero-shot teacher inference

Choose a few representative samples and run the proposed teacher path **before any optimizer or EMA update**.

For models with native editing capability, start by using it to convey the reference or target information. Clearly identify the role of each reference image in the teacher instruction. Other context designs are welcome; choose something that the model can meaningfully consume.

For each example, compare normal conditioning with the proposed teacher conditioning. **Only change the conditioning. Keep everything else in inference unchanged:**

- The initial model weights, output size, sampler, sampling step count, and actual timestep/sigma values.
- The native guidance setting and other generation options.
- The random seed and, where supported, the same initial generation noise.
- The preprocessing rules for shared inputs.

If conditioning preparation consumes random numbers, use a separate generator or save the initial generation noise so that the comparison remains controlled. Extra teacher inputs necessarily add conditioning computation, but should not silently change the generation schedule or output dimensions.

Save the teacher context, processed reference images, and resulting images. Assess the property you want to teach: subject fidelity, style, editing success, or another task-specific behavior. Also check for framing changes, unwanted copying, and failure to follow the request.

If the teacher is unhelpful, first inspect or revise the context and preprocessing. Increasing the teacher's sampling steps or CFG would change this diagnostic; keep the native settings while assessing the context itself.

These zero-shot images are a practical way to evaluate the proposed context. During D-OPSD training, the teacher is evaluated on **student-generated states**. Once that path is connected, inspecting its clean-latent predictions can help diagnose any gap between the standalone teacher results and the actual training signal.

## 4. Implement the model-specific interfaces

Keep the core training loop easy to inspect. The following responsibilities can be separate helpers or clearly separated code blocks; they do not require a new framework:

| Responsibility | Required behavior |
| --- | --- |
| Student condition | Build exactly the inputs used during normal inference |
| Teacher condition | Add the chosen privileged information through supported conditioning interfaces |
| Prediction | Run the appropriate branch at the shared student state and step |
| Output selection | Select the generated latent positions when reference tokens are also present |
| Clean-latent conversion | Convert the model's prediction using its own time/sigma and parameterization conventions |
| Sampler step | Advance the generated state using the student prediction and native solver behavior |

For flow models, check the velocity sign and time direction before implementing the conversion. The expressions `x_t - t*v` and `x_t + (1-t)*v` apply to different conventions. Use the target model's definition, including any required output transformation.

For sequence-based editing models, teacher and student may have different numbers of conditioning tokens. Their predictions must still align over the **same generated image positions**. Check attention masks and position/reference IDs as well as tensor shapes.

Keep teacher-only information out of student conditioning and caches. In particular, a prompt rewritten using the target image is itself privileged information. Any native prompt expansion used by the student should depend only on its normal inference inputs.

## 5. Connect the training update

Use the [core pseudocode](README.md#the-core-procedure) and [recommended settings](recipes.md).

At each rollout step, detach the generated state, evaluate both branches on that state, compute predicted `x_0` MSE, and advance with the student prediction. Average the loss across the rollout steps. Teacher predictions do not receive gradients, and state transitions do not introduce backpropagation through the full trajectory.

Use EMA decay **0.9999** by default:

```text
teacher = 0.9999 * teacher + 0.0001 * student
```

Apply this after an actual student optimizer update. A fixed teacher keeps its initial parameters throughout training.

For LoRA, the public examples share frozen base weights and maintain student and teacher adapters. Verify that adapter switching and gradient checkpoint recomputation use the intended adapter, and that the optimizer updates only student parameters.

For full fine-tuning, maintain separate teacher parameter state for all trained student parameters. Updating LoRA tensors alone is no longer sufficient, and a frozen-teacher forward must not accidentally use already-updated student weights. The public training entry points currently implement LoRA and leave full fine-tuning to be added; see [issue #11](https://github.com/vvvvvjdy/D-OPSD/issues/11).

## 6. Adapt memory use and distributed execution

The public examples use a straightforward DeepSpeed ZeRO-2 setup. Choose the execution framework according to the target model and hardware; it is separate from the D-OPSD objective.

| Main memory cost | Options to consider |
| --- | --- |
| Frozen encoders and VAE | Precompute conditions/latents when their inputs and preprocessing are fixed; offload modules when idle |
| Activations | Smaller micro-batches, gradient accumulation, gradient checkpointing, and attention implementations supported by the model |
| Model parameters and optimizer state | FSDP2 or ZeRO-3, and CPU offload where appropriate |
| Long image/video token sequences | Sequence/context parallelism if the model and training stack support it |

ZeRO-2 partitions optimizer states and gradients; it does not partition model parameters. Adding data-parallel GPUs therefore may not solve a parameter-memory bottleneck. See [DeepSpeed ZeRO](https://deepspeed.readthedocs.io/en/latest/zero3.html) and [PyTorch FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html).

The detached step transitions also allow per-step backward accumulation as a memory optimization. Preserve the average-loss scaling and defer the optimizer/EMA update until the intended update boundary; confirm compatibility with the distributed engine.

When changing frameworks, adapt teacher state, EMA updates, gradient accumulation, and checkpoint save/load together. A full fine-tuning teacher also consumes parameter memory. Cache only computations that remain valid: activations depending on updated model weights cannot be treated as fixed dataset features.

## 7. Validate a small run and hand off the result

Before a longer run, establish that:

- The adapted student sampler reproduces the native baseline within expected numerical differences when weights and inputs are unchanged.
- Both branches use the same generated state and time; teacher-only inputs remain confined to the teacher.
- Loss and gradients are finite, intended student parameters change, and teacher changes match the chosen EMA/fixed policy.
- A short training run produces interpretable progress on fixed examples, including requests beyond exact training examples.
- Saved student weights or LoRA reload successfully in a fresh native inference pipeline.

Compare the original and trained student with the same normal inputs, seeds, resolution, and native sampling settings. Assess both the desired improvement and generation quality. For editing, also inspect preservation of content that should remain unchanged.

The final handoff should contain the adapted training entry point and launch configuration, data-field mapping, teacher-context description, representative comparisons, and a student loading example. Record the model/library versions and actual sampling schedule. If training is intended to resume, also verify restoration of the teacher, optimizer, and relevant training state.

**The trained student can keep the original inference workflow completely unchanged apart from loading its new weights or LoRA.** No teacher branch or teacher-only target information is needed at deployment.

For unresolved behavior or a new context idea, use [the troubleshooting notes](troubleshooting.md) and join the discussion in [GitHub Issues](https://github.com/vvvvvjdy/D-OPSD/issues).
