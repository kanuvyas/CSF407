# Second-order CPT  P(next | previous-two-words)  (observed contexts only)

| context (X_{t-2}, X_{t-1}) | distribution of X_t |
|---|---|
| (<START>, <START>) | the: 1.000 |
| (<START>, the) | cat: 0.500, dog: 0.500 |
| (cat, ran) | to: 1.000 |
| (cat, sat) | on: 1.000 |
| (dog, ran) | to: 1.000 |
| (dog, sat) | on: 1.000 |
| (on, the) | mat: 0.500, rug: 0.500 |
| (ran, to) | the: 1.000 |
| (sat, on) | the: 1.000 |
| (the, cat) | ran: 0.333, sat: 0.667 |
| (the, dog) | ran: 0.333, sat: 0.667 |
| (the, mat) | <END>: 1.000 |
| (the, park) | <END>: 1.000 |
| (the, rug) | <END>: 1.000 |
| (to, the) | park: 1.000 |