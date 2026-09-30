"""maps.py - the ASCII maps used by the lab (Tasks 0-3) so that the run script and
the tests use exactly the same inputs."""

WAREHOUSE = """
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################
"""

# Test 2: goal directly next to the start.
TRIVIAL = """
#####
#SG##
#####
"""

# Test 3: goal sealed off by walls (the wall at column 4 of the last row).
NO_SOLUTION = """
#######
#S....#
###.###
#...#G#
#######
"""

# Test 4a: two routes; top route = 4 moves, bottom route = 8 moves.
ALTERNATIVE = """
#######
#S...G#
#.###.#
#.....#
#######
"""

# Test 4b: open room, many routes of equal length (4 moves).
OPEN_ROOM = """
#####
#S..#
#...#
#..G#
#####
"""

# Extra (not in the handout): a warehouse with open floor and two shelf blocks.
# Used to see A* vs BFS on a map where the heuristic is not misleading.
OPEN_WAREHOUSE = """
#################
#S..............#
#..###....###...#
#..#........#...#
#..#........#...#
#..###....###...#
#...............#
#..............G#
#################
"""

# Extra: found by a seeded random search (see REPORT.md, Task 6). Shortest path is 6 moves
# (Right, Right, Up, Right, Right, Up). Heuristics that overestimate are lured up column 1
# and return 8 moves.
HEURISTIC_TRAP = """
#######
#...#G#
#.#...#
#S...##
#######
"""
