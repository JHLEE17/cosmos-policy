# Cosmos Policy Architecture Analysis

> Scope: this document analyzes the official `Cosmos Policy` paper and the current public code in this repo.
> Goal: provide a flow-chart-ready description of model architecture, latent layout, and inference phases, with concrete dimensions.
> Key takeaway: Cosmos Policy does **not** add a separate action head. It reuses the pretrained `Cosmos-Predict2-2B-Video2World` latent video diffusion model and treats `action`, `proprio`, `future state`, and `value` as special latent-frame slots inside the video latent sequence.

---

## 1. Executive Summary

Cosmos Policy is a fine-tuned `Cosmos-Predict2-2B-Video2World` diffusion transformer policy. The base model is a latent video diffusion model:

- vision tokenizer: `Wan2.1`-style spatiotemporal VAE
- latent shape per frame: `16 x 28 x 28` for `224 x 224` inputs
- backbone: DiT-like transformer with hidden size `2048`, `28` blocks, `16` heads
- language condition: T5 embeddings with shape `(B, 512, 1024)`

The central trick is **latent frame injection**:

1. build a pseudo-video sequence made of:
   - one blank frame
   - current camera observations
   - blank placeholders for proprio/action/value
   - placeholders for future observations
2. VAE-encode that sequence into latent frames
3. overwrite selected latent frames with duplicated low-dimensional signals:
   - current proprio
   - action chunk
   - future proprio
   - value
4. train / sample the model as if it were denoising a video latent sequence
5. read the action back from the designated action latent frame

This means that "the model watches a video and outputs an action" is only partially true. In the released public inference path, the model does **not** ingest a real temporal history clip. Instead, it ingests a **structured multi-slot pseudo-video** built from the current observation and blank placeholders, then generates the missing slots.

---

## 2. Source Map Used For This Analysis

Paper and appendix:

- `docs/[Cosmos-Policy] 2601.16163v1.pdf`

Main architecture and policy wrappers:

- `cosmos_policy/models/policy_text2world_model.py`
- `cosmos_policy/models/policy_video2world_model.py`
- `cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py`
- `cosmos_policy/config/conditioner/video2world_conditioner.py`

Tokenizer and base diffusion stack:

- `cosmos_policy/tokenizers/wan2pt1.py`
- `cosmos_policy/_src/predict2/models/text2world_model.py`
- `cosmos_policy/_src/predict2/utils/model_loader.py`

Backbone network:

- `cosmos_policy/_src/predict2/configs/video2world/defaults/net.py`
- `cosmos_policy/_src/predict2/networks/minimal_v1_lvg_dit.py`
- `cosmos_policy/_src/predict2/networks/minimal_v4_dit.py`

Public inference entrypoints:

- `cosmos_policy/experiments/robot/cosmos_utils.py`
- `README.md`
- `cosmos_policy/experiments/robot/libero/run_libero_eval.py`
- `cosmos_policy/experiments/robot/robocasa/run_robocasa_eval.py`
- `cosmos_policy/experiments/robot/aloha/run_aloha_eval.py`

Sampling and noise:

- `cosmos_policy/modules/cosmos_sampler.py`
- `cosmos_policy/modules/hybrid_edm_sde.py`

---

## 3. Base Model Stack

### 3.1 Top-level inheritance

The public policy model is not a separate architecture from scratch. The class hierarchy is:

```text
ImaginaireModel
  -> DiffusionModel
     -> CosmosPolicyDiffusionModel
        -> CosmosPolicyVideo2WorldModel
```

Relevant files:

- base diffusion scaffold: `cosmos_policy/_src/predict2/models/text2world_model.py`
- policy-specific latent injection and loss masking: `cosmos_policy/models/policy_text2world_model.py`
- video-conditioning and inference conditioning logic: `cosmos_policy/models/policy_video2world_model.py`

### 3.2 Backbone

The released public checkpoints use the `cosmos_v1_2B` network registered in:

- `cosmos_policy/_src/predict2/configs/video2world/defaults/net.py`

For the `2B` backbone:

| Item | Value |
| --- | --- |
| Network class | `MinimalV1LVGDiT` |
| Hidden size | `2048` |
| Transformer blocks | `28` |
| Attention heads | `16` |
| Head dim | `128` |
| Input latent channels | `16` |
| Output latent channels | `16` |
| Spatial patch size | `2` |
| Temporal patch size | `1` |
| Cross-attn text dim | `1024` |
| Positional embedding | `rope3d` |

### 3.3 Effective input channel count into patch embedding

There are two channel augmentations before patch embedding:

1. `MinimalV1LVGDiT` adds `+1` channel for the `condition_video_input_mask`
2. `MiniTrainDIT` adds `+1` channel for the `padding_mask` when `concat_padding_mask=True`

Therefore, while the latent itself has `16` channels, the patch embed layer effectively sees:

- `16 + 1 + 1 = 18` channels

Important nuance:

- model output still predicts `16` latent channels
- the extra channels are only conditioning signals for the transformer input

---

## 4. Tokenizer And Latent Geometry

### 4.1 VAE / tokenizer

The repo uses `Wan2pt1VAEInterface` in:

- `cosmos_policy/tokenizers/wan2pt1.py`

Key constants:

| Item | Value |
| --- | --- |
| Latent channels | `16` |
| Spatial compression | `8` |
| Temporal compression | `4` |
| `get_latent_num_frames(num_pixel_frames)` | `1 + (num_pixel_frames - 1) // 4` |

### 4.2 Pixel-to-latent dimension conversion

For public policy inference:

- input images are resized to `224 x 224`
- latent height = `224 / 8 = 28`
- latent width = `224 / 8 = 28`
- latent channels = `16`

So one latent frame has:

- shape: `(16, 28, 28)`
- number of elements: `16 x 28 x 28 = 12,544`

### 4.3 Temporal compression detail

The first frame is treated specially:

- frame 0 is encoded without temporal compression
- the remaining frames are temporally compressed in groups of `4`

That is why the code builds:

- one leading blank frame
- then **four identical copies** for each semantic slot that should become one latent frame

This matches the paper appendix explanation:

- blank first image -> blank first latent
- all later semantic slots are created by four duplicated images -> one latent frame each

---

## 5. Latent Frame Injection: The Core Mechanism

### 5.1 Concept

Instead of adding:

- an action decoder head
- a proprio encoder branch
- a separate value head

Cosmos Policy simply overwrites chosen latent frames with duplicated low-dimensional signals.

The three main injectors in code are:

- `replace_latent_with_action_chunk(...)`
- `replace_latent_with_proprio(...)`
- direct fill for value frame in `compute_loss_with_epsilon_and_sigma(...)`

### 5.2 Action injection math

Suppose:

- latent frame size = `16 x 28 x 28 = 12,544`
- action chunk shape = `(K, d_act)`

Then:

1. flatten action chunk into `(K * d_act,)`
2. repeat it enough times to fill `12,544` values
3. truncate if needed
4. reshape to `(16, 28, 28)`
5. overwrite the designated action latent frame

This is exactly what `replace_latent_with_action_chunk(...)` does.

### 5.3 Proprio injection math

Current and future proprio use the same logic:

1. start from vector shape `(d_prop,)`
2. repeat to fill `12,544` elements
3. reshape to `(16, 28, 28)`
4. overwrite the selected latent frame

### 5.4 Value injection math

Value is simpler:

1. value is a scalar
2. scalar is broadcast to all `12,544` elements in the value latent frame

### 5.5 Extraction at inference time

#### Action extraction

`extract_action_chunk_from_latent_sequence(...)`:

1. take the generated action latent frame `(16, 28, 28)`
2. flatten to `(12,544,)`
3. reshape the prefix into repeated chunks of shape `(K * d_act,)`
4. average across repeated copies
5. reshape back to `(K, d_act)`
6. optionally unnormalize from `[-1, 1]` to dataset action scale

Important implementation nuance:

- for `LIBERO` and `RoboCasa`, `12,544` is an exact multiple of `K * d_act`
- for `ALOHA`, it is **not** exact:
  - `K * d_act = 50 * 14 = 700`
  - `12,544 // 700 = 17`
  - so extraction averages the first `17` full copies and ignores the remaining `644` tail elements

That detail is not central to the paper, but it is true in the current public implementation.

#### Value extraction

`extract_value_from_latent_sequence(...)`:

1. take the value latent frame
2. flatten
3. average every element
4. map from `[-1, 1]` back to `[0, 1]`

#### Future image extraction

Future images are different:

- they are actual VAE-decoded image-like latent frames
- before decoding, the code restores injected non-image slots to the original clean placeholder latents to avoid visual artifacts

This "undo latent injection before decode" step is implemented in:

- `undo_latent_injection(...)`
- `get_future_images_from_generated_samples(...)`

Important nuance:

- actions and values do **not** require VAE decode
- only future image visualization does

---

## 6. The Latent Sequence Layout

This is the single most important part for any flow chart.

### 6.1 General semantic template

The policy arranges the latent sequence as:

```text
[blank] [current state slots] [action] [future state slots] [value]
```

This ordering is intentionally:

- left-to-right
- compatible with autoregressive decoding
- semantically close to `(s, a, s', V(s'))`

### 6.2 LIBERO layout

Configured in `cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py`:

| Latent index | Meaning |
| --- | --- |
| 0 | blank |
| 1 | current proprio |
| 2 | current wrist image |
| 3 | current primary image |
| 4 | action chunk |
| 5 | future proprio |
| 6 | future wrist image |
| 7 | future primary image |
| 8 | value |

Suite constants:

| Item | Value |
| --- | --- |
| `state_t` | `9` |
| `chunk_duration` | `33` |
| `min_num_conditional_frames` | `4` |
| action dim | `7` |
| proprio dim | `9` |
| chunk size | `16` |

Raw-to-latent:

- raw input frames: `33`
- latent frames: `1 + (33 - 1) // 4 = 9`

### 6.3 RoboCasa layout

| Latent index | Meaning |
| --- | --- |
| 0 | blank |
| 1 | current proprio |
| 2 | current wrist image |
| 3 | current primary image |
| 4 | current secondary image |
| 5 | action chunk |
| 6 | future proprio |
| 7 | future wrist image |
| 8 | future primary image |
| 9 | future secondary image |
| 10 | value |

Suite constants:

| Item | Value |
| --- | --- |
| `state_t` | `11` |
| `chunk_duration` | `41` |
| `min_num_conditional_frames` | `5` |
| action dim | `7` |
| proprio dim | `9` |
| chunk size | `32` |

### 6.4 ALOHA layout

| Latent index | Meaning |
| --- | --- |
| 0 | blank |
| 1 | current proprio |
| 2 | current left wrist image |
| 3 | current right wrist image |
| 4 | current primary image |
| 5 | action chunk |
| 6 | future proprio |
| 7 | future left wrist image |
| 8 | future right wrist image |
| 9 | future primary image |
| 10 | value |

Suite constants:

| Item | Value |
| --- | --- |
| `state_t` | `11` |
| `chunk_duration` | `41` |
| `min_num_conditional_frames` | `5` |
| action dim | `14` |
| proprio dim | `14` |
| chunk size | `50` |

### 6.5 One subtle but important point

In the paper and config, future proprio is part of the modeled future state. In the public helper path:

- the latent slot for future proprio exists
- the model is trained to predict it
- but `cosmos_utils.get_future_state_prediction(...)` only exposes future image predictions publicly

So the model architecture supports future proprio prediction, but the main helper API does not currently return it as a user-facing field.

---

## 7. What The Model Actually Receives At Inference

### 7.1 It is a pseudo-video, not a real observation clip

The main public inference function is:

- `cosmos_policy/experiments/robot/cosmos_utils.py::get_action(...)`

It builds a structured input sequence from **one current observation**:

- current camera images
- current proprio
- blank placeholders
- duplicated future placeholders

This means:

- the public base inference path is **single-step observation based**
- it does **not** use temporal history by default
- it uses the video backbone as a structured multi-slot latent generator

This matches the paper statement that the model uses:

- observation at `t`
- predicts observation at `t + K`
- does not model long observed history in the base setup

### 7.2 Raw pixel input shapes

For batch size `B`:

- video tensor shape before VAE encode: `(B, 3, T_raw, 224, 224)`

Where:

- `LIBERO`: `T_raw = 33`
- `RoboCasa`: `T_raw = 41`
- `ALOHA`: `T_raw = 41`

### 7.3 Latent tensor shapes

After VAE encode:

- shape: `(B, 16, T_latent, 28, 28)`

Where:

- `LIBERO`: `(B, 16, 9, 28, 28)`
- `RoboCasa`: `(B, 16, 11, 28, 28)`
- `ALOHA`: `(B, 16, 11, 28, 28)`

---

## 8. Conditioning Mechanism

### 8.1 Video conditioning mask

The conditioning object stores:

- `gt_frames`
- `condition_video_input_mask_B_C_T_H_W`

Mask shape:

- `(B, 1, T_latent, 28, 28)`

Semantics:

- `1` -> this latent frame is treated as clean conditioning
- `0` -> this latent frame is a target to denoise / generate

### 8.2 Default direct-policy conditioning

For public direct inference, `num_conditional_frames` defaults to:

- `4` for `LIBERO`
- `5` for `RoboCasa`
- `5` for `ALOHA`

That means:

- all current-state slots are conditioning
- action, future state, and value slots are generated

Example for LIBERO:

```text
mask = [1, 1, 1, 1, 0, 0, 0, 0, 0]
```

### 8.3 World model conditioning

For future-state prediction:

- the code sets `num_conditional_frames = min_num_conditional_frames + 1`
- so the action frame also becomes conditioning

Meaning:

- direct policy: predict `(a, s', V)`
- world model phase: condition on `(s, a)` and generate `(s', V)`

### 8.4 Value function conditioning

For value prediction:

- code sets `num_conditional_frames = state_t - 1`
- all frames except the final value slot are conditioning

Then optional masking chooses between:

- `V(s')`: mask current state and action, condition mostly on future state
- `Q(s,a)`: mask future state, condition on current state and action

This is implemented through mask edits in:

- `CosmosPolicyVideo2WorldModel.get_data_and_condition(...)`
- `CosmosPolicyVideo2WorldModel.get_x0_fn_from_batch(...)`

---

## 9. Direct Policy Inference Flow

This is the fastest and most important public path.

### 9.1 Stage A: load model

`get_model(cfg)`:

1. resolve checkpoint path
2. call `load_model_from_checkpoint(...)`
3. instantiate the Hydra config
4. build tokenizer, conditioner, net
5. load checkpoint weights

### 9.2 Stage B: encode language

Task instruction is represented as T5 embeddings:

- shape: `(1, 512, 1024)` before batch repeat
- batched shape: `(B, 512, 1024)`

Source:

- `cosmos_policy/_src/predict2/inference/get_t5_emb.py`

### 9.3 Stage C: preprocess observation

`get_action(...)`:

1. choose camera order depending on suite
2. resize every image to `224 x 224`
3. optional JPEG compression
4. optional test-time image transforms if the checkpoint was trained with augmentation
5. normalize proprio if configured

### 9.4 Stage D: build raw pseudo-video sequence

For each suite, `get_action(...)` appends:

1. blank first frame
2. placeholder for current proprio
3. current wrist / wrists
4. current third-person image / images
5. placeholder for action
6. placeholder for future proprio
7. placeholders for future images
8. placeholder for value

Every semantic slot after the first is represented by `4` identical images so that the VAE makes one latent frame per slot.

### 9.5 Stage E: VAE encode to latent sequence

The raw pseudo-video is passed as:

- `data_batch["video"]`

and encoded into:

- `x0: (B, 16, T_latent, 28, 28)`

### 9.6 Stage F: prepare clean conditioning latents

During `get_x0_fn_from_batch(...)`:

1. set video condition mask
2. inject current proprio into the current-proprio slot
3. keep current observation slots as clean conditioning
4. action / future / value slots remain generative targets

### 9.7 Stage G: diffusion sampling

Sampling is done by:

- `CosmosPolicyDiffusionModel.generate_samples_from_batch(...)`
- `CosmosPolicySampler`

Default public settings:

| Suite | Action denoising steps |
| --- | --- |
| LIBERO | `5` |
| RoboCasa | `5` |
| ALOHA | `10` |

Noise range in inference-only configs:

- `sigma_max = 80`
- `sigma_min = 4`

Sampler behavior nuance:

- if `num_steps > 1`, sampler effectively runs solver steps and then one final clean denoise
- if `num_steps == 1`, it directly denoises once

### 9.8 Stage H: network forward dimensions

Take `LIBERO` as example:

- latent input: `(B, 16, 9, 28, 28)`
- effective patch-embed input channels: `18`
- patch size: spatial `2`, temporal `1`
- patch grid per frame: `14 x 14 = 196`
- token count: `9 x 196 = 1764`
- hidden size per token: `2048`

For `RoboCasa / ALOHA`:

- token count: `11 x 196 = 2156`

The backbone does:

1. patch embed
2. 3D RoPE positional embedding
3. timestep embedding from sigma
4. 28 transformer blocks with:
   - self-attention
   - cross-attention to text
   - MLP
5. final layer
6. unpatchify back to latent shape `(B, 16, T_latent, 28, 28)`

### 9.9 Stage I: action extraction

After denoising:

1. select action latent frame at `action_latent_idx`
2. flatten and average repeated copies
3. reshape to `(chunk_size, action_dim)`
4. unnormalize to dataset action scale if configured

That final action chunk is what gets executed.

### 9.10 Stage J: optional future-state and value extraction in parallel mode

If `generate_future_state_and_value_in_parallel=True`, the same generated sample is also used to:

- decode future images
- extract value

This is still "parallel decoding" because:

- action, future state, and value are sampled from one joint denoising pass

Important nuance in implementation:

- the helper extracts value from latent index `-1`, which works because the value slot is the last latent frame

---

## 10. Autoregressive Planning Inference Flow

This is the slower but more accurate planning path.

### 10.1 Two-model deployment

The paper and code distinguish:

- **policy model**: original Cosmos Policy checkpoint
- **planning model**: rollout-refined checkpoint used as world model and value function

This is called "dual deployment" in the paper.

### 10.2 Phase breakdown

Planning is split into three main inference phases:

1. **Action proposal generation**
2. **Future state prediction**
3. **Value prediction**

Then:

4. **proposal ranking / selection**

### 10.3 Phase 1: action proposal generation

For each proposal:

- run the normal direct policy action generation
- optionally across multiple GPUs in parallel for best-of-N

Paper example:

- best-of-`N`
- choose highest-value action after planning rollouts

Repo support:

- `num_queries_best_of_n`
- parallel workers across multiple GPUs

### 10.4 Phase 2: future state prediction

`get_future_state_prediction(...)`:

1. reuse the previously generated latent from the action phase
2. set `num_conditional_frames = current_conditioning + 1`
3. treat action slot as conditioning
4. sample future-state slots autoregressively

Important code detail:

- this uses `skip_vae_encoding=True`
- `previous_generated_latent=generated_latent_with_action`

So the model does **not** rebuild the latent from scratch. It reuses the action sample latent and continues generation conditioned on it.

### 10.5 Phase 3: value prediction

`get_value_prediction(...)`:

1. take future-state sample(s)
2. set `num_conditional_frames = state_t - 1`
3. only leave value slot as target
4. optionally apply masking for `V(s')` or `Q(s,a)`
5. sample value slot autoregressively

Again it reuses latents:

- `skip_vae_encoding=True`
- `previous_generated_latent=fs_sample`

### 10.6 Ensembles

The paper describes:

- 3 world-model predictions per action
- 5 value predictions per future state
- total 15 value predictions per action

The repo exposes the same concept:

- `use_ensemble_future_state_predictions`
- `num_future_state_predictions_in_ensemble`
- `use_ensemble_value_predictions`
- `num_value_predictions_in_ensemble`

### 10.7 Value aggregation

The helper supports several schemes:

- `average`
- `lcb`
- `success_vote`
- `majority_mean`

The paper specifically highlights `majority mean` for robustness under multimodal / high-variance values.

### 10.8 Search depth > 1

The public repo also supports a deeper search tree beyond the basic one-step best-of-N:

- `search_depth > 1`

Implementation idea:

1. predict action
2. predict future state
3. replace current-state slots with predicted future-state slots
4. predict the next action
5. repeat for deeper lookahead

This is a real implementation feature in the current repo and is useful to know for a flow chart, even though the paper's core explanation is usually framed as one action chunk plus imagined future state/value.

---

## 11. Training Objectives And Masking

### 11.1 Three roles in one model

The unified model serves as:

- policy: `p(a, s', V(s') | s)`
- world model: `p(s', V(s') | s, a)`
- value function: `p(V(s') | s, a, s')`

### 11.2 Training batch split described by paper

The paper describes:

- 50% demo samples for policy training
- 50% rollout samples, split equally into:
  - world model training
  - value function training

The code implements the corresponding masks:

- `rollout_data_mask`
- `world_model_sample_mask`
- `value_function_sample_mask`

### 11.3 Loss masking

`compute_loss_with_epsilon_and_sigma(...)` supports selective frame loss:

- demo samples: emphasize action prediction, optionally with future-state/value auxiliary targets
- rollout world-model samples: emphasize future-state slots
- rollout value-function samples: emphasize value slot

The important point is:

- the model always predicts a full latent sequence
- training masks decide **which slots count toward loss**

So role specialization happens through conditioning masks and loss masks, not through architectural branching.

---

## 12. Noise Schedule And Sampler Behavior

### 12.1 Training-time sigma distribution

The paper appendix and code match here:

- 70% from original log-normal-like EDM distribution
- 30% from uniform high-sigma range `[1.0, 85.0]`

In code:

- `HybridEDMSDE.sample_t(...)`

This is intended to improve action prediction quality at high-noise early denoising stages.

### 12.2 Experiment config values

Typical training config in released experiments:

- `sigma_max = 200`
- `sigma_min = 0.01`

Inference-only override:

- `sigma_max = 80`
- `sigma_min = 4`

The paper explicitly notes that raising inference `sigma_min` improves policy accuracy.

---

## 13. Exact Dimension Reference

### 13.1 Shared dimensions

| Item | Value |
| --- | --- |
| Image resolution | `224 x 224` |
| Latent channels | `16` |
| Latent spatial size | `28 x 28` |
| Elements per latent frame | `12,544` |
| Text tokens | `512` |
| Text embedding dim | `1024` |
| Backbone hidden dim | `2048` |
| Transformer blocks | `28` |
| Attention heads | `16` |
| Head dim | `128` |
| Patch size | `2 x 2 x 1` |

### 13.2 Suite-specific dimensions

| Item | LIBERO | RoboCasa | ALOHA |
| --- | --- | --- | --- |
| latent frames `state_t` | `9` | `11` | `11` |
| raw pseudo-video frames | `33` | `41` | `41` |
| conditional latent frames | `4` | `5` | `5` |
| action chunk size | `16` | `32` | `50` |
| action dim | `7` | `7` | `14` |
| proprio dim | `9` | `9` | `14` |
| action elements | `112` | `224` | `700` |
| exact copies in action latent during extraction | `112` | `56` | `17` full copies |
| token count after patching | `1764` | `2156` | `2156` |

### 13.3 Patch token count formula

Given latent shape `(B, 16, T, 28, 28)` and patch sizes `(1, 2, 2)`:

- temporal patches per frame group = `T`
- height patches = `28 / 2 = 14`
- width patches = `28 / 2 = 14`
- sequence length = `T x 14 x 14`

Therefore:

- `LIBERO`: `9 x 14 x 14 = 1764`
- `RoboCasa / ALOHA`: `11 x 14 x 14 = 2156`

---

## 14. What "Action From Video" Really Means Here

For flow-chart purposes, the most accurate phrasing is:

> Cosmos Policy uses a pretrained **video latent diffusion backbone** to generate a structured latent sequence whose designated action slot is decoded into an action chunk.

More concretely:

- the model does not decode action from pixels with a separate regression head
- the model does not first generate future video and then use an inverse dynamics model
- the model does not output discrete action tokens

Instead:

1. current observation images are encoded to video latents
2. proprio/action/value are represented as latent-frame slots
3. the diffusion model denoises the whole sequence
4. the action chunk is read directly from the action latent slot

That is why the paper emphasizes:

- no architectural modification
- no separate action module
- single-stage fine-tuning

---

## 15. Flow-Chart Ready Phase Decomposition

If you later draw a flow chart, I recommend these nodes.

### 15.1 Direct policy chart nodes

1. `Observation`
2. `Image preprocessing`
3. `Task text -> T5 embedding`
4. `Pseudo-video assembly`
5. `VAE encode -> latent sequence`
6. `Inject proprio into latent slot`
7. `Build conditioning mask`
8. `Diffusion sampling with DiT backbone`
9. `Extract action latent slot -> action chunk`
10. `Optional: decode future image slots`
11. `Optional: extract value slot`
12. `Execute action chunk`

### 15.2 Planning chart nodes

1. `Policy model: sample N action proposals`
2. `Planning model: condition on (s, a) and predict future state`
3. `Planning model: condition on future-state sample and predict value`
4. `Aggregate ensemble values`
5. `Select best action`
6. `Execute selected action`

### 15.3 Optional deeper-search nodes

1. `Replace current-state slots with predicted future-state slots`
2. `Predict next action`
3. `Predict next imagined future state`
4. `Predict next value`
5. `Aggregate across search depth`

---

## 16. Final Interpretation

The current official repo implements Cosmos Policy as a **unified latent video diffusion policy**:

- pretrained video model backbone
- no new action head
- multimodal slot injection in latent space
- one model reused as policy, world model, and value function

Inference is divided into two practical modes:

1. **direct policy / parallel decoding**
   - one denoising pass predicts action, future state, and value together
   - only action is needed for execution

2. **planning / autoregressive decoding**
   - action first
   - then future state conditioned on action
   - then value conditioned on future state or masked variants
   - choose the best proposal via best-of-N search

For a future flow chart, the most important structural axes are:

- raw observation -> pseudo-video -> latent sequence
- latent slot semantics
- conditioning mask changes across phases
- direct parallel decoding vs autoregressive planning decoding
- action extraction directly from the latent slot

