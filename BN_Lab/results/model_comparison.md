Comparison of first-order and second-order models
(vocabulary: 10 words; context tokens |V_ctx| = 11 incl. <START>; output tokens |V_out| = 11 incl. <END>)

| metric | first-order | second-order |
|---|---|---|
| possible contexts (|V_ctx|^n) | 11 | 121 |
| observed contexts | 11 | 15 |
| unobserved (zero-probability) contexts | 0 | 106 (87.6%) |
| full CPT size (contexts x |V_out|) | 121 | 1331 |
| distinct non-zero parameters | 17 | 19 |
| zero entries inside the full CPT | 104 | 1312 |
| distinct sentences in 2000 samples | 292 | 6 |
| of which NOT in training data | 286 | 0 |
| samples hitting max length (20) | 71 | 0 |
| mean sentence length (words) | 6.07 | 6.00 |

Example novel sentences (first-order): ['the mat', 'the rug', 'the park', 'the cat ran to the mat', 'the cat ran to the rug', 'the dog ran to the mat']
Example novel sentences (second-order): []

Sentence probabilities (chain rule):
  P('the cat sat on the mat')  first-order = 0.02778   second-order = 0.16667
  P('the cat sat on the park')  first-order = 0.02778   second-order = 0.00000
  P('the dog ran to the mat')  first-order = 0.01389   second-order = 0.00000
