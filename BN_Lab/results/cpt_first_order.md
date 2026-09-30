# First-order CPT  P(next | current)

Full table (rows = current word, columns = next word; 0 = zero-probability transition)

| current \ next | cat | dog | mat | on | park | ran | rug | sat | the | to | <END> |
|---|---|---|---|---|---|---|---|---|---|---|---|
| <START> | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| cat | 0 | 0 | 0 | 0 | 0 | 0.333 | 0 | 0.667 | 0 | 0 | 0 |
| dog | 0 | 0 | 0 | 0 | 0 | 0.333 | 0 | 0.667 | 0 | 0 | 0 |
| mat | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| on | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| park | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| ran | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| rug | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| sat | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| the | 0.25 | 0.25 | 0.167 | 0 | 0.167 | 0 | 0.167 | 0 | 0 | 0 | 0 |
| to | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |

## Required words: distributions and zero-probability transitions

**P(next | the)**: cat: 0.250, dog: 0.250, mat: 0.167, park: 0.167, rug: 0.167  
zero-probability next words: on, ran, sat, the, to, <END>

**P(next | cat)**: ran: 0.333, sat: 0.667  
zero-probability next words: cat, dog, mat, on, park, rug, the, to, <END>

**P(next | dog)**: ran: 0.333, sat: 0.667  
zero-probability next words: cat, dog, mat, on, park, rug, the, to, <END>

**P(next | sat)**: on: 1.000  
zero-probability next words: cat, dog, mat, park, ran, rug, sat, the, to, <END>

**P(next | ran)**: to: 1.000  
zero-probability next words: cat, dog, mat, on, park, ran, rug, sat, the, <END>
