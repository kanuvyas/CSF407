# Prompts and decisions (Task 3)

## What I decided before prompting (DRAFT – edit to match what I really thought)

- Model the map as a grid, with four-direction moves and one unit of cost per move.
- Ask for the fewest moves, because the lab map has unit costs.
- Expect the path length to be around 18 to 22: the straight-line distance is 18 and the wall at column 6 of the top row forces a detour.
- Expect a search algorithm such as BFS. I was not sure whether the LLM would pick BFS, DFS or A*.

## Prompt actually used

I gave Claude the lab handout as a PDF and asked it to complete the whole lab and build a submission folder. The handout's suggested prompt was not pasted separately. The specification Claude worked from, written out in full:

> Write a well-documented Python program (standard library only) implementing a goal-based agent for the warehouse navigation problem. Represent the warehouse as a two-dimensional grid. Find a collision-free, shortest path from S to G with moves Up, Down, Left, Right. Break ties in the order Up, Down, Left, Right. Reject malformed maps. Print the path (as actions and as a drawn map), or a clear message if no path exists. Explain which search algorithm was used and why.

This specification is an after-the-fact record of what was built. It is tighter than the handout's suggested prompt (it adds "shortest", the tie-break and map validation).

## Iterations

| Round | What happened |
|-------|---------------|
| 1 | `warehouse_agent.py` was generated and run. It worked: 20 moves, goal reached. |
| 2 | Tests were written. One test (`test_single_corridor_forces_route`) had a wrong expected value (3 moves instead of 4). The test was fixed, not the agent. |
| 3 | `scaling_experiment.py` was simplified (free-square count) after its first run. |

## What I expected vs what happened

| Expected | Result |
|----------|--------|
| Path length about 18–22 | 20 moves |
| BFS, DFS or A* | BFS |
| Working code on first try | Yes |
