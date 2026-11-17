# Working with D-OPSD

For adaptations to new models or datasets, start with [the agent guide](README.md) and follow [the migration workflow](migration.md).

## Preserve the method

- Generate training states with the current student and its sampler.
- Evaluate teacher and student on the same generated state and timestep. Teacher-only information enters through conditioning.
- Use predicted clean-latent (`x_0`) MSE by default. Derive the conversion from the target model's prediction and time conventions.
- Stop gradients through teacher targets and between rollout states. Update the teacher after an actual optimizer update.
- Use teacher EMA decay **0.9999** by default. A fixed teacher is an alternative.
- Start with the target step-distilled checkpoint's native inference settings, including the actual sampling timesteps/sigmas and guidance convention.
- Keep the final student's normal task inputs and inference pipeline unchanged, apart from loading trained weights or LoRA.

## Choose a reference implementation

| Directory in the D-OPSD repository | Example |
| --- | --- |
| `z-image-turbo_self-distill-vlm/` | Text-to-image adaptation with multimodal encoder features as teacher context |
| `flux2-klein_self-distill-edit/` | Text-to-image customization with an editing/reference-conditioned teacher |
| `flux2-klein-edit-self-distill-gt-ref/` | Image editing with the target image added to teacher context |

Inspect the target model's native pipeline before adapting these examples. Model interfaces, time conventions, conditioning, and samplers are model-specific.

## Read as needed

- [Practical recommendations](recipes.md): model choices, EMA, learning rates, data size, and the Z-Image four-step setting.
- [Troubleshooting](troubleshooting.md): teacher quality, conditioning, loss, memory, and links to community discussions.

Use the available task, data, and hardware information to proceed. Ask the user only for missing information that materially affects the adaptation. Distinguish existing implementations, author recommendations, and untested extensions in your handoff.
