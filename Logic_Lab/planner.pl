% Warehouse knowledge base (Task 6)
connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

can_move(X,Y) :-
    connected(X,Y).

% Task 7: checking moves proposed by the Python planner
valid_move(X,Y) :-
    connected(X,Y).

% small extra: check a whole path of locations, e.g. valid_path([a,b,c])
valid_path([_]).
valid_path([X,Y|T]) :-
    valid_move(X,Y),
    valid_path([Y|T]).

% Task 8: road example
wet_road.
slippery :- wet_road.
reduce_speed :- slippery.

% Task 0/Task 6 style: facts vs rules demo from the handout
penguin(polly).
bird(X) :- penguin(X).
animal(X) :- bird(X).
