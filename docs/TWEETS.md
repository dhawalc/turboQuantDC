# Tweet Thread -- TurboQuantDC 5.1x Result

## Tweet 1 (hook)

5.1x KV cache compression matching FP16 generation quality. 3-bit keys + 1-bit residual signs + 2-bit values + FP16 window. Every other TurboQuant implementation gets garbled output at 3-bit. Root cause was a one-line protocol bug in get_mask_sizes that broke the attention mask. Thread on how I found it.

github.com/dhawalc/turboQuantDC

## Tweet 2 (autoresearch)

I built an autoresearch loop. Load model once, sweep 600 configurations overnight, auto-score each on 8 test prompts. It evolved the algorithm from "0/5 prompts correct" at round 0 to "matches FP16 on all 8" by round 6. The winning config was not in the TurboQuant paper. It was a novel approach called ResidualQuant -- store actual residual signs instead of random projection signs.

## Tweet 3 (the bug)

The root cause of EVERY generation failure was get_mask_sizes returning query_length instead of cached_length + query_length. One line. Attention metrics look perfect. Generation produces garbage. The vLLM maintainer who got 0% gsm8k accuracy with TurboQuant-3 probably hit the same bug. Every custom HF Cache implementation should check this.

## Tweet 4 (ResidualQuant)

QJL (the paper's 1-bit correction stage) is unbiased but high-variance. ResidualQuant stores signs of the actual residual instead of random-projection signs. Same bit budget. But it preserves the residual direction perfectly instead of destroying it through a random Gaussian projection. Every team working on TurboQuant has independently confirmed QJL hurts generation. ResidualQuant fixes it at zero extra cost.

@karpathy @no_stp_on_snek @GoogleResearch

## Tweet 5 (CTA)

Full details in the repo. 568+ tests, 21 source modules, 40+ AI agents, MIT license. The autoresearch dashboard is live. Paper: arxiv 2504.19874.

github.com/dhawalc/turboQuantDC
pip install turboquantdc
