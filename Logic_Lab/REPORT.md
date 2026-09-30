# Laboratory Report: Logical Reasoning for Planning

Logic + Search = Planning. All results below are produced by `run_all.sh` (logs in `results/`).

---

## Task 0: The planning problem

**(a) Initial state:** I = {At(Robot,A), At(Package,A)}
**(b) Goal:** G = {At(Package,C)}

**(c)/(d) Actions, preconditions and effects** (locations A, B, C; A–B and B–C connected in both directions):

| Action | Preconditions | Effects |
|---|---|---|
| Move(x,y) for (A,B), (B,A), (B,C), (C,B) | At(Robot,x) | ¬At(Robot,x), At(Robot,y) |
| PickUp(Package,l) | At(Robot,l), At(Package,l) | ¬At(Package,l), Holding(Package) |
| Drop(Package,l) | At(Robot,l), Holding(Package) | ¬Holding(Package), At(Package,l) |

The handout only gives `PickUp(Package,A)` and `Drop(Package,C)` as examples; I assumed PickUp and Drop exist at all three locations.

**Is PickUp(Package,A) applicable in I?** Yes. Its preconditions are At(Robot,A) and At(Package,A), and both are in I, so I ⊨ Preconditions.

**Is Drop(Package,C) applicable in I?** No. It needs At(Robot,C) and Holding(Package), and neither is in I (the robot is at A and is not holding anything). Being in the list of actions is not enough; all preconditions must hold in the current state.

---

## Task 1: Manual plan

| State | Facts |
|---|---|
| S0 | At(Robot,A), At(Package,A) |
| S1 (after PickUp(Package,A)) | At(Robot,A), Holding(Package) |
| S2 (after Move(A,B)) | At(Robot,B), Holding(Package) |
| S3 (after Move(B,C)) | At(Robot,C), Holding(Package) |
| S4 (after Drop(Package,C)) | At(Robot,C), At(Package,C) |

S4 ⊨ G, so this is a valid plan of length 4.

**Note on the handout's example:** the handout suggests the sequence Move(A,B), PickUp(Package,B), Move(B,C), Drop(Package,C). That sequence is **not** valid: the package is still at A when the robot is at B, so PickUp(Package,B) needs At(Package,B), which is false. The robot has to pick up the package *before* leaving A. I checked this with the validator (see Extra test below).

---

## Task 2: LLM prompt and code

The prompt is in `prompts.md` and the code is `planner.py`. Where the ideas from the specification appear:

* **Preconditions → when is an action applicable?** `applicable(state, action)`: `action.pos_pre <= state and not (action.neg_pre & state)`. This is the check S ⊨ Preconditions(a).
* **Effects → how does the state change?** `apply_action`: `(state - neg_eff) | pos_eff` (remove negative effects, then add positive effects).
* **Goal → when does planning terminate?** `if goal <= nxt: return new_path`, which tests that every goal fact is in the state (it is also checked on the initial state).
* **BFS → how are alternative plans explored?** A `deque` used as a FIFO queue with a `visited` set: all plans of length k are expanded before any plan of length k+1, so the first plan found is a shortest one. If the queue empties, the function returns `None`, which is printed as "No plan found".

Assumptions: states are sets of ground propositions (closed-world: anything not in the set is false), actions are deterministic, and there are no costs (BFS minimises the number of actions).

---

## Task 3: Tests

| Test | Initial state | Goal | Plan found? | Plan | Actually valid? |
|---|---|---|---|---|---|
| **A**: original problem | At(Robot,A), At(Package,A) | At(Package,C) | Yes | PickUp(Package,A), Move(A,B), Move(B,C), Drop(Package,C) | Yes: validator checked all 4 steps and the goal |
| **B**: PickUp removed | same | At(Package,C) | No: "No plan found" | none | n/a (nothing invented) |
| **C1**: robot-only goal | same | At(Robot,C) | Yes | Move(A,B), Move(B,C) | Yes for this goal, but the package is still at A, so At(Package,C) is **not** satisfied |
| **C2**: PickUp removed, package goal | same | At(Package,C) | No: "No plan found" | none | n/a |

* **Test A:** the planner's plan is identical to my manual plan and the independent validator replays it with every precondition satisfied.
* **Test B:** without PickUp nothing can ever make At(Package,C) true (Drop needs Holding(Package), and only PickUp creates it), so the planner correctly exhausts the search and reports "No plan found" instead of inventing an action.
* **Test C:** I interpreted "an action that moves the robot but not the package" as the Move actions themselves. In C1 the robot reaches C, but the final state still contains At(Package,A) and does not contain At(Package,C), so the robot reaching C is not treated as the package reaching C. In C2 the robot can reach C, but the planner still reports no plan for the package goal.

**Extra tests (also in `tests/test_planner.py`):**
* The handout's example sequence is rejected by the validator at step 2: *PickUp(Package,B) NOT applicable (missing At(Package,B))*.
* With the precondition check switched off, the planner "solves" the problem with the single action `Drop(Package,C)`, producing a state that contains both At(Package,A) and At(Package,C). The validator rejects it because At(Robot,C) and Holding(Package) were missing.

All 6 pytest tests pass (`results/pytest.txt`).

---

## Task 4: Logic and search

Completed flow:

```
Current state
   ↓
Check action preconditions
   ↓
Is S ⊨ Preconditions(a)?  (yes: a is applicable; no: skip it)
   ↓
Generate successor state S' = Apply(S, a)
   ↓
Search over alternatives (BFS puts S' on the queue with the plan so far)
   ↓
Goal? (G ⊆ S'?  yes: return the plan; no: keep expanding the queue)
```

**In my own words:** The logical part looks at one state and one action and answers "is this action allowed here, and what does the world look like afterwards?" (`applicable` and `apply_action`). It doesn't decide which action to do. The search part takes those answers and decides in what order to explore the possibilities: BFS keeps a queue of states and tries every applicable action from each, level by level, until a state satisfies the goal. Without logic, search wouldn't know which moves are legal or what they change; without search, logic would only verify one action at a time and never find a sequence. **Logic determines what is possible; search determines what to try.**

---

## Task 5 (optional): Can the LLM verify its own plan?

An LLM-style explanation of the plan, written in the requested form:

> PickUp(Package,A): needs At(Robot,A) and At(Package,A); both hold in S0. → S1 has Holding(Package), no At(Package,A).
> Move(A,B): needs At(Robot,A); holds in S1. → S2 has At(Robot,B).
> Move(B,C): needs At(Robot,B); holds in S2. → S3 has At(Robot,C).
> Drop(Package,C): needs At(Robot,C) and Holding(Package); both hold in S3. → S4 has At(Package,C) = goal.

I compared it with the state transitions printed by the validator (`results/planner_output.txt`): the states after each step are the same as the explanation says, so here the explanation happens to be correct.

**Which should I trust more? (b) the independently executed state transitions.** The executed transitions are computed mechanically from the action definitions: the code can't "forget" a precondition or assume a fact that isn't in the state. An LLM explanation is text that sounds convincing and is produced without actually checking anything, so it can contain a wrong claim (for example, it could say At(Package,B) holds when the robot is at B, the same mistake that is in the handout's example plan). Its explanation is useful for understanding, but only the executed check counts as verification: **a generated explanation is not an independent verification.**

---

## "Think About It" answers (summary)

* **Applicability:** an action is applicable only if all its preconditions are satisfied in the current state (Task 0).
* **Where the spec shows up in the code:** see the four bullets in Task 2.
* **Logic + search:** logic decides what is possible, search decides what to try (Task 4).

---

## Reflection Questions

1. **Why specify preconditions and effects before asking the LLM?** They are the definition of the problem. If I don't write them down, the LLM will choose its own (possibly different) version and the code could solve a different problem, and I would have no reference to check it against. With a precise specification I can also tell whether the generated code matches it.
2. **Error without checking preconditions:** the planner would accept `Drop(Package,C)` in the initial state and "deliver" the package instantly without ever picking it up or moving (I demonstrated this: plan = [Drop(Package,C)], with At(Package,A) and At(Package,C) both true).
3. **Why is a plan that looks reasonable not necessarily valid?** Because validity depends on every precondition being true in the exact state where each action runs, which can't be seen at a glance. The handout's example (Move(A,B), PickUp(Package,B), …) looks natural but fails because the package is still at A.
4. **What did the LLM contribute?** The boilerplate implementation: the action representation, the BFS loop, printing, the test scaffolding and the Prolog file. It also produced the explanation in Task 5.
5. **What did I have to verify independently?** That the preconditions and effects in the code match the specification, that the plan is valid (I used a separate validator and replayed the plan by hand), that the planner says "No plan found" when it should, that the robot's location isn't confused with the package's location, and that the handout's own example plan is invalid.
6. **Where is logical reasoning used?** In deciding applicability (S ⊨ Preconditions(a)), in computing the effects of an action on the state, in the goal test (S ⊨ G), and in the optional Prolog part, where facts and rules are used to infer whether a move is supported.
7. **How is planning related to search?** Planning is search in a state space: states are nodes, applicable actions are edges, the initial state is the start, and any state satisfying G is a goal. BFS here is the same uninformed search studied earlier (it finds a shortest plan, at the cost of memory); the difference is that the successor function comes from logical preconditions and effects instead of a given graph.

---

## Optional extension: Prolog

The knowledge base is in `planner.pl`; queries were run with SWI-Prolog 9.0.4 (`results/prolog_output.txt`).

### Task 6

| Query | Result |
|---|---|
| `?- can_move(a,b).` | true |
| `?- can_move(a,c).` | false |

**(a) Why true for can_move(a,b)?** There is a fact `connected(a,b).` and the rule `can_move(X,Y) :- connected(X,Y).` so with X = a and Y = b the body is a known fact, hence the head follows.
**(b) Why not can_move(a,c)?** There is no fact `connected(a,c)`, and no other rule can derive it, so Prolog cannot prove it. (Prolog answers "false" because it can't prove it from the knowledge base, which is not the same as saying it is false in the real world; this is the closed-world assumption.) It also matches the warehouse: A and C are not directly connected, the robot has to go through B.
**(c) Relationship to Connected(X,Y) → CanMove(X,Y):** the rule `can_move(X,Y) :- connected(X,Y).` is this implication written backwards (head :- body means "head if body"), with X and Y universally quantified.

### Task 7

| Query | Result |
|---|---|
| `valid_move(a,b)` | true |
| `valid_move(b,c)` | true |
| `valid_move(a,c)` | false |
| (extra) `valid_path([a,b,c])` | true |
| (extra) `valid_path([a,c])` | false |

**Challenge:** if the Python planner proposed `Move(a,c)`, Prolog would answer false for `valid_move(a,c)`, meaning this action is not supported by the warehouse knowledge, so the proposal would be rejected. (My Python planner would never propose it, because it only has the four Move actions given in the specification.)

### Task 8

`?- reduce_speed.` returns **true**. Prolog tries the rule `reduce_speed :- slippery.`, which needs `slippery`; that needs `wet_road`, which is a fact, so the goal is proven.

Wet_road (fact) ⇒ (Wet_road → Slippery) ⇒ Slippery ⇒ (Slippery → ReduceSpeed) ⇒ ReduceSpeed.

### Prolog reflection

1. **Fact vs rule:** a fact is something stated to be true unconditionally (`connected(a,b).`); a rule says something is true if other conditions hold (`can_move(X,Y) :- connected(X,Y).`).
2. **Query = entailment question:** a query asks whether the statement follows from the facts and rules; Prolog answers true if it can derive it and false if it cannot.
3. **Why verify a Python plan with Prolog?** The Prolog knowledge base is written separately from the planner, so it checks the plan against the stated facts of the warehouse rather than against the planner's own assumptions. A bug in the planner is unlikely to be repeated in the Prolog rules.
4. **Advantage of an independent verifier for LLM-generated plans:** the LLM's code and explanation can look right and still be wrong. A separate verifier built from a small, human-readable set of facts and rules doesn't share the LLM's mistakes and is easy to check by eye. Note that it only checks what the knowledge base contains: my Prolog rules check movement connectivity, not preconditions such as Holding(Package), so they are a partial check only.
