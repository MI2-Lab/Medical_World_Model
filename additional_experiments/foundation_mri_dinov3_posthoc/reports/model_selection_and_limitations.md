# DINOv3 post-hoc model selection and limitations

## Selection

The sole new encoder is `facebook/dinov3-vitb16-pretrain-lvd1689m` at
Hugging Face revision `5931719e67bbdb9737e363e781fb0c67687896bc`.
ViT-B/16 was selected because its 85,660,416 parameters, 768-dimensional
tokens, patch size 16, and 12-block architecture closely match the already
published DINO v1 ViT-B/16 reference. This isolates the pretraining-generation
change more cleanly than a larger DINOv3 model would.

The checkpoint is read from an existing local Hugging Face snapshot in strict
offline mode. The experiment verifies the 342,662,192-byte safetensors file
against SHA-256
`9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`.
It does not download, copy, commit, or redistribute the checkpoint.

## Frozen representation

The image adapter is byte-for-byte inherited from the published DINO v1
baseline: early, late, and late-minus-pre DCE channels; 32 axial slices;
224-pixel input; identical GLOBAL and fixed central 64-mm LOCAL geometry;
identical clipping and ImageNet normalization.

At 224 pixels, DINOv3 produces one CLS token, four register tokens, and 196
patch tokens. Each slice representation is the final normalized CLS token
concatenated with the mean of the 196 patch tokens. Register tokens are
explicitly excluded. The 32 slice vectors are averaged, giving 1,536 values
per visit and spatial axis.

## Why this is not a new preregistered formal candidate

This extension was requested after the original outcomes and final report had
already been published. It is therefore post-hoc and descriptive.

DINOv3 uses a custom license rather than Apache-2.0 or another standard open
source license. Local use relies on the already available authorized snapshot;
the experiment does not purport to accept terms on behalf of an institution.

The LVD-1689M corpus was curated from a non-enumerable public web-image pool.
Official documentation does not disclose direct use of I-SPY raw MRI, but it
also provides no patient-level membership list or I-SPY exclusion manifest.
Consequently, zero I-SPY-derived pretraining contamination cannot be proven.
All results must be described as an unknown-contamination sensitivity analysis,
not a contamination-free replacement for the original candidate set.

Official sources:

- <https://github.com/facebookresearch/dinov3>
- <https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/MODEL_CARD.md>
- <https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/LICENSE.md>
- <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m/tree/5931719e67bbdb9737e363e781fb0c67687896bc>

