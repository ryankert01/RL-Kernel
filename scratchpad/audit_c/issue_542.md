PR #444 implemented `final_logit_softcap` on the `test-gemma` branch, but the operator is not yet available on `main`. The computation is reusable across models, so I propose moving it into the shared operator directories and merging it into `main`.

Two models currently covered by the project need this computation:

| Model | Related issue | Softcap value | Usage |
| --- | --- | --- | --- |
| Gemma-4-31B-it | #415 | `30` | Applies softcap to the final logits |
| Muse-Glimmer-30B | #501 | `20` | Applies `output_multiplier` before softcap |

Both use the same softcap definition:

$$
y=c\tanh(x/c)
$$

Proposed changes, based on #444:

1. Move the implementation, PyTorch reference, tests, and documentation into the shared operator layout.
2. Replace the fixed value of `30` with a `softcap` parameter, keeping `30` as the default at the public interface.
3. Add forward and backward correctness tests and training/inference consistency tests for `softcap=20` and `softcap=30`.
4. Add usage documentation and performance comparisons for the shared interface.

Muse-Glimmer would apply `output_multiplier` in the model code before calling the shared operator. The shared operator would only implement the parameterized softcap computation.

