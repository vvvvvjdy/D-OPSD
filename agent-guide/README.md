# D-OPSD: A Migration Guide for Coding Agents

This guide helps you adapt [D-OPSD](https://github.com/vvvvvjdy/D-OPSD) to a new model, task, or dataset. It is written for coding agents and the people working with them.

**Preserve on-policy self-distillation, design useful teacher context for the task, and keep the trained student's native inference behavior.** The existing implementations demonstrate this procedure with different ways of constructing teacher context.

## Reading order

| Document | Purpose |
| --- | --- |
| This page | Understand the method and choose a reference implementation |
| [Migration workflow](migration.md) | Inspect a new model, adapt data and conditioning, implement training, and validate the result |
| [Practical recommendations](recipes.md) | Use our recommended starting settings and understand model-specific examples |
| [Troubleshooting](troubleshooting.md) | Diagnose common problems using implementation checks and existing issue discussions |

## Give this guide to your coding agent

Ask your agent to read this page and the [agent instructions](AGENTS.md), then follow the migration workflow. You can use the following request; paths are relative to the D-OPSD repository root:

```text
Read agent-guide/README.md and agent-guide/AGENTS.md, then follow
agent-guide/migration.md to adapt D-OPSD to my setup.

Target model/checkpoint: <model ID or local path>
Task and desired improvement: <subject, style, editing behavior, domain, etc.>
Dataset: <path and a brief description of its fields>
Training mode: <LoRA or full fine-tuning>
Hardware: <GPU count and memory per GPU>

Inspect the target model's native inference implementation and the relevant
D-OPSD example. Preserve student-generated trajectories and teacher supervision
on the same states. Start with the model's native sampling settings, predicted
x_0 MSE, and teacher EMA decay 0.9999.

First run a few zero-shot teacher examples by changing only the conditioning;
keep the other inference settings unchanged. Then implement and validate a small
training run, and provide a way to load the trained student using the original
inference pipeline. Ask me about missing task-specific information when needed.
```

## What D-OPSD is useful for

Step-distilled diffusion models generate images in a small number of sampling steps. Users often want to teach them a subject, style, editing behavior, or domain preference using a small collection of examples, while retaining their few-step generation capability.

D-OPSD uses the training data to provide richer conditioning to a teacher. The student learns from the teacher's predictions on the student's own generated states. This provides supervision from paired data without requiring an external reward function. See the [paper](https://arxiv.org/abs/2605.05204) for the method and experiments.

After training, load the student weights or LoRA into the original pipeline. The inference code, normal task inputs, sampling schedule, and guidance settings can remain unchanged. For a text-to-image student, inference can remain text-only. For an editing student, it continues to receive the source image(s) and editing instruction. Teacher-only target information is used during training.

## The core procedure

Let `q` be the input available during normal inference, and `r` be additional information available during training. Construct student condition `c_s` from `q`, and teacher condition `c_T` from `q` and `r`.

Initialize the student and teacher from the same step-distilled model. The teacher is an EMA copy by default, with decay **0.9999**; a fixed teacher is also worth trying.

At a state visited by the current student's sampler, evaluate both branches:

$$
v_s = v_\theta(x_t^s,t,c_s), \qquad
v_T = v_{\bar\theta}(x_t^s,t,c_T).
$$

Both predictions are made on the **same student state** and at the **same time**. The next rollout state is computed with the student prediction and the chosen native sampler.

Our recommended objective follows the public code: for a rollout with `K` steps, compare the predicted clean latents,

$$
\mathcal L = \frac{1}{K}\sum_{k=1}^{K}
\operatorname{MSE}\!\left(
\hat x_{0,s}^{(k)},\operatorname{sg}(\hat x_{0,T}^{(k)})
\right).
$$

Here `x_0` denotes the clean latent, and `sg` denotes stop-gradient. Derive each clean prediction from the target model's own parameterization. With the flow convention `x_0 = x_t - t v`, this objective equals `t² MSE(v_s, v_T)` at a shared state. The reverse time convention used in the Z-Image training example gives `(1-t)² MSE(v_s, v_T)`. This is a time-weighted alignment of teacher and student predictions. The clean prediction is not the dataset's GT latent. See [issue #6](https://github.com/vvvvvjdy/D-OPSD/issues/6).

The following is conceptual pseudocode; helper names are not repository APIs:

```text
initialize student and teacher from the same step-distilled checkpoint

for each training batch:
    c_s = construct normal task conditioning
    c_T = construct conditioning with teacher-only information
    x = sample native initial noise
    loss = 0

    for each step in the native sampling schedule:
        state = stop_gradient(x)
        pred_T = teacher(state, step, c_T), without gradients
        pred_s = student(state, step, c_s)

        clean_T = convert_prediction_to_clean(state, step, pred_T)
        clean_s = convert_prediction_to_clean(state, step, pred_s)
        loss += MSE(clean_s, stop_gradient(clean_T)) / number_of_steps

        x = stop_gradient(native_sampler_step(state, step, pred_s))

    backpropagate loss and update student
    update teacher with EMA after the optimizer update
```

For gradient accumulation, perform the EMA update after each actual optimizer update. For a fixed teacher, omit its EMA update. The sampler helper includes any native solver state and step-dependent behavior required by the model.

## Teacher context is an open design space

The teacher should have useful information about the desired output that the student does not receive during normal inference. Its usefulness depends on how the model interprets that information.

These are existing examples:

| Task | Student conditioning | Teacher conditioning | Reference implementation |
| --- | --- | --- | --- |
| Text-to-image customization with multimodal encoder features | Text | Text and target-image information encoded into compatible multimodal features | [Z-Image example](https://github.com/vvvvvjdy/D-OPSD/tree/main/z-image-turbo_self-distill-vlm) |
| Text-to-image subject/style customization using editing capability | Text | Text, a reference image, and a teacher instruction | [Klein customization example](https://github.com/vvvvvjdy/D-OPSD/tree/main/flux2-klein_self-distill-edit) |
| Image editing | Source image(s) and editing instruction | Source image(s), the target image, and an instruction identifying their roles | [Klein editing example](https://github.com/vvvvvjdy/D-OPSD/tree/main/flux2-klein-edit-self-distill-gt-ref) |

For models with native editing/reference-image support, we recommend starting with that capability when constructing the teacher. Extra text, multiple references, other supported modalities, or a different organization of context may also be useful. These examples do not exhaust the possible designs.

Before training, inspect a few **zero-shot teacher inference results**. Change the conditioning to include the teacher context while keeping the other inference settings unchanged: checkpoint, sampling steps and time values, solver, guidance, output resolution, and random seed/initial noise. This helps you decide whether the context provides useful guidance. See the [workflow](migration.md#3-check-zero-shot-teacher-inference).

We welcome new ideas, adaptations, results, and questions in [GitHub Issues](https://github.com/vvvvvjdy/D-OPSD/issues). Include the model, task, context construction, and representative examples so that others can understand and build on your findings.
