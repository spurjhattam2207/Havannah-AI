import time
import math
import random
import numpy as np
from helper import *
from copy import deepcopy

def get_virtual_neighbours(state: np.array, move: Tuple[int, int]):
        i,j = move
        siz = state.shape[0]//2
        dim = state.shape[0]
        neighbors = []
        if j < siz-1:
            neighbors.append(((i-2,j-1),(i-1,j),(i-1,j-1)))
            neighbors.append(((i-1,j-2),(i-1,j-1),(i,j-1)))
            neighbors.append(((i+1,j-1),(i,j-1),(i+1,j)))
            neighbors.append(((i+2,j+1),(i+1,j),(i+1,j+1)))
            neighbors.append(((i+1,j+2),(i+1,j+1),(i,j+1)))
            neighbors.append(((i-1,j+1),(i,j+1),(i-1,j)))
        elif j == siz-1: 
            neighbors.append(((i-2,j-1), (i-1,j-1), (i-1,j)))
            neighbors.append(((i-1,j-2),(i-1,j-1),(i,j-1)))
            neighbors.append(((i+1,j-1),(i,j-1),(i+1,j)))
            neighbors.append(((i+2,j+1),(i+1,j),(i+1,j+1)))
            neighbors.append(((i,j+2),(i,j+1),(i+1,j+1)))
            neighbors.append(((i-1,j+1),(i,j+1),(i-1,j)))
        elif j == siz: 
            neighbors.append(((i-2,j-1), (i-1,j-1), (i-1,j)))
            neighbors.append(((i-1,j-2),(i-1,j-1),(i,j-1)))
            neighbors.append(((i+1,j-1),(i,j-1),(i+1,j)))
            neighbors.append(((i+1,j+1),(i+1,j),(i,j+1)))
            neighbors.append(((i-1,j+2),(i-1,j+1),(i,j+1)))
            neighbors.append(((i-2,j+1),(i-1,j),(i-1,j+1)))
        elif j == siz+1:
            neighbors.append(((i-1,j-1),(i,j-1),(i-1,j)))
            neighbors.append(((i,j-2),(i,j-1),(i+1,j-1)))
            neighbors.append(((i+2,j-1),(i+1,j),(i+1,j-1)))
            neighbors.append(((i+1,j+1),(i+1,j),(i,j+1)))
            neighbors.append(((i-1,j+2),(i-1,j+1),(i,j+1)))
            neighbors.append(((i-2,j+1),(i-1,j),(i-1,j+1)))
        else:
            neighbors.append(((i-1,j-1),(i,j-1),(i-1,j)))
            neighbors.append(((i+1,j-2),(i+1,j-1),(i,j-1)))
            neighbors.append(((i+2,j-1),(i+1,j),(i+1,j-1)))
            neighbors.append(((i+1,j+1),(i+1,j),(i,j+1)))
            neighbors.append(((i-1,j+2),(i-1,j+1),(i,j+1)))
            neighbors.append(((i-2,j+1),(i-1,j),(i-1,j+1)))
        
        valid_virtual_neighbours = [] 
        for n in neighbors:
            (x,y) = n[0]
            if x < 0 or x > dim-1 or y < 0 or y > dim-1:
                continue
            if state[x,y] == 3:
                continue
            valid_virtual_neighbours.append(n)
        return valid_virtual_neighbours

def check_virtual_neighbours(state, move, v_neighbour, v_n1, v_n2):
    x, y = move
    nx, ny = v_neighbour
    if state[x, y] == state[nx, ny] and state[v_n1[0], v_n1[1]] == 0 and state[v_n2[0], v_n2[1]] == 0:
        return True
    return False 

def find_parent(dsu_dict: Dict[Tuple[int, int], Tuple[Tuple[int, int], int]], key: Tuple[int, int]) -> Tuple[int, int]:
    if dsu_dict[key][0] == key:
        return key, dsu_dict[key][1]
    else:
        dsu_dict.update({key: find_parent(dsu_dict, dsu_dict[key][0])})
        return dsu_dict[key]

def union(dsu_dict: Dict[Tuple[int, int], Tuple[Tuple[int, int], int]], key1: Tuple[int, int], key2: Tuple[int, int]) -> None:
    parent1 = find_parent(dsu_dict, key1)
    parent2 = find_parent(dsu_dict, key2)

    if parent1[0] != parent2[0]:
        if parent2[1] >= parent1[1]:
            dsu_dict.update({parent1[0]: (parent2[0], parent1[1] + parent2[1])})
            dsu_dict.update({parent2[0]: (parent2[0], parent1[1] + parent2[1])})
        else:
            dsu_dict.update({parent1[0]: (parent1[0], parent1[1] + parent2[1])})
            dsu_dict.update({parent2[0]: (parent1[0], parent1[1] + parent2[1])})

def update_virtual_neighbours(state: np.array, move: Tuple[int, int], dsu_dict: Dict[Tuple[int, int], Tuple[Tuple[int, int], int]]):
    find_parent(dsu_dict, move)
    virtual_added = False
    map_v_pairs = get_virtual_neighbours(state, move)
    for v in map_v_pairs:
        if len(v) == 0: continue
        v2 = v[0]
        if check_virtual_neighbours(state, move, v[0], v[1], v[2]):
            union(dsu_dict, move, v2)
            virtual_added = True

    return virtual_added

def update_dsu(state: np.array, move: Tuple[int, int], dsu_dict: Dict[Tuple[int, int], Tuple[Tuple[int, int], int]], vc: bool = True) -> Dict[Tuple[int, int], Tuple[Tuple[int, int], int]]:
    for (nx, ny) in get_neighbours(state.shape[0], move):
        if state[nx, ny] == state[move[0], move[1]]:
            union(dsu_dict, (nx, ny), move)
    if vc: update_virtual_neighbours(state, move, dsu_dict)

def create_dict(state: np.array) -> Dict[Tuple[int, int], Tuple[Tuple[int, int], int]]:
    n, _ = state.shape
    dsu_dict = {}

    for i in range(n):
        for j in range(n):
            if state[i][j] == 1 or state[i][j] == 2:
                dsu_dict[(i, j)] = ((i, j), 1)

    for i in range(n):
        for j in range(n):
            if state[i][j] == 1 or state[i][j] == 2:
                current_move = (i, j)
                neighbours = get_neighbours(n, current_move)

                for (nx, ny) in neighbours:
                    if (nx, ny) in dsu_dict and state[nx][ny] == state[i][j]:
                        union(dsu_dict, (i, j), (nx, ny))
            else:
                dsu_dict[(i, j)] = ((i, j), 1)

    return dsu_dict

def check_win_own(current_state: np.array, dict_of_dsu: Dict[Tuple[int, int], Tuple[Tuple[int, int], int]], move: Tuple[int, int]):
    dim = int(math.sqrt(len(dict_of_dsu)))
    p = find_parent(dict_of_dsu, move)
    
    def check_bridge_own():
        flag = 0
        corners = get_all_corners(dim)
        for corner in corners:
            if find_parent(dict_of_dsu, corner) == p:
                flag += 1
            if flag == 2: return True
        return False
    
    value = check_bridge_own()
    
    if value: return True
    
    def check_fork_own():
        flag = 0
        edges = get_all_edges(dim)
        for edge in edges:
            for indi in edge:
                if find_parent(dict_of_dsu, indi) == p:
                    flag += 1
                    break
            if flag == 3: return True
        return False
    
    value = check_fork_own()
    
    if value: return True
    
    return check_ring(current_state == current_state[move], move)

def heuristic(given_state: np.array, move: List[Tuple[int, int]], player: int, dsu: Tuple[Tuple[int, int], int]) -> float:
    dim = given_state.shape[0]
    if move is None: return 0
    neighbours = get_neighbours(dim, move)
    edges = get_all_edges(dim)

    def create_virtual_connections() -> int:
        return update_virtual_neighbours(given_state, move, dsu)

    def maintain_virtual_connections() -> float:
        neighbours = get_neighbours(dim, move)
        count = 0
        for neighbour in neighbours:
            if given_state[neighbour[0], neighbour[1]] == player or given_state[neighbour[0], neighbour[1]] == 0:
                continue
            else:
                useful = set()
                neighbours_of_neighbour = get_neighbours(dim, neighbour)
                if len(neighbours) != 4:
                    for n in neighbours_of_neighbour:
                        if n in neighbours and given_state[n[0], n[1]] == player:
                            useful.add(n)
                    if len(useful) == 2:
                        count += 1
                else:
                    for n in neighbours_of_neighbour:
                        if n in neighbours and given_state[n[0], n[1]] == player:
                            useful.add(n)
                    if len(useful) == 1:
                        count += 0.2
        return count

    
    def increment_in_group_size() -> int:
        neighbours = get_neighbours(dim, move)
        parents_seen = set()
        sizes_seen = []
        for neighbour in neighbours:
            if given_state[neighbour[0], neighbour[1]] == player:
                parent = find_parent(dsu, neighbour)
                if parent[0] not in parents_seen:
                    sizes_seen.append(parent[1])
                    parents_seen.add(parent[0])
        
        if len(sizes_seen) == 0: return 0
        return np.sum(sizes_seen) + 1 - np.max(sizes_seen)
    
    def vc_with_edge():
        neighbours = {x: True for x in get_neighbours(dim, move)}
        for edge_set in edges:
            common = 0
            for edge in edge_set:
                common += int(neighbours.get(edge, False))
            if move in edge_set: return 1.2
            if common >= 2: return 1
        return 0

    return 10 * maintain_virtual_connections() + 0.5 * create_virtual_connections() + 0.07 * increment_in_group_size() + 0.1 * vc_with_edge()

class MCTSNode:
    def __init__(self, state, dsu, player=1, parent=None, move=None, actions=set(), pnum=1):
        self.state = state
        self.dsu = dsu
        self.parent = parent
        self.move = move
        self.visits = 0
        self.wins = 0
        self.children = []
        self.valid_actions = actions
        self.untried_moves = list(actions)
        self.k = (2 * (3 - player == pnum) - 1) * heuristic(state, move, 3 - player, dsu)
        random.shuffle(self.untried_moves)
        self.player = player
        self.rave_wins = np.zeros_like(state) 
        self.rave_visits = np.zeros_like(state)
        self.player_number = pnum
        
    def __repr__(self):
        return f"Player {self.player} LastMove {self.move} Visits: {self.visits} Wins: {self.wins} NumChildren: {len(self.children)} UntriedMoves: {len(self.untried_moves)}"

    def expand(self):
        move = self.untried_moves.pop()
        next_state = self.state.copy()
        next_state[move[0], move[1]] = self.player
        tmp = deepcopy(self.valid_actions)
        tmp.remove(move)
        dsu_copy = deepcopy(self.dsu)
        update_dsu(next_state, move, dsu_copy)
        child_node = MCTSNode(next_state, dsu_copy, 3 - self.player, self, move, actions=tmp, pnum=self.player_number)
        self.children.append(child_node)
        return child_node
    
    def tree_policy(self, weight=2, k=50):
        curr_node = self
        while not self.is_terminal_state(self.state, self.dsu, self.move):
            if not curr_node.is_fully_expanded():
                return curr_node.expand()
            curr_node = curr_node.best_child_rave(weight, k)
            if curr_node is None: break
        return curr_node

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, exploration_weight):
        values = [(child.wins / child.visits) + exploration_weight * \
                        math.sqrt(math.log(self.visits) / child.visits) for child in self.children]
        return self.children[np.argmax(values)]

    def best_child_rave(self, exploration_weight=1.41, k=50):
        best_score = float('-inf')
        best_children = []

        for child in self.children:
            uct_score = (child.wins / child.visits) + exploration_weight * \
                        math.sqrt(math.log(self.visits) / child.visits)

            rave_score = 0
            if np.sum(child.rave_visits) > 0:
                rave_score = (np.sum(child.rave_wins) / np.sum(child.rave_visits))

            if child.visits == 0: knowledge_score = 0
            else: knowledge_score = child.k / np.sqrt(child.visits)

            rave_weight = k / (k + child.visits)
            total_score = (1 - rave_weight) * uct_score + rave_weight * rave_score + knowledge_score

            if total_score > best_score:
                best_score = total_score
                best_children = [child]
            elif total_score == best_score:
                best_children.append(child)

        return random.choice(best_children) if best_children else None

    def backpropagate(self, result, visited_moves=None):
        self.visits += 1
        self.wins += result

        if visited_moves is not None:    
            self.rave_visits += visited_moves
            if result == 1: self.rave_wins += visited_moves

        if self.parent:
            self.parent.backpropagate(result, visited_moves)

    def rollout(self):
        curr_state = self.state.copy()
        curr_player = self.player
        curr_move = self.move
        visited_moves = np.zeros_like(curr_state)
        valid_moves = deepcopy(list(self.valid_actions))
        dsu_copy = deepcopy(self.dsu)

        while not self.is_terminal_state(curr_state, dsu_copy, curr_move):
            if len(valid_moves) == 0: break
            curr_move = random.choice(valid_moves)
            x, y = curr_move
            curr_state[x, y] = curr_player
            update_dsu(curr_state, curr_move, dsu_copy)
            curr_player = 3 - curr_player
            visited_moves[x, y] = 1
            valid_moves.remove(curr_move)

        return self.evaluate_game(curr_state, dsu_copy, curr_move), visited_moves

    def is_terminal_state(self, state, dsu, move):
        if move is None:
            return False
        return np.sum(state == 0) == 0 or check_win_own(state, dsu, move)

    def evaluate_game(self, state, dsu, move):
        if check_win_own(state, dsu, move):
            if state[move[0], move[1]] == self.player_number:
                return 1
            else:
                return 0
        return 0.5

class AIPlayer:
    def __init__(self, player_number: int, timer):
        """
        Intitialize the AIPlayer Agent

        # Parameters
        `player_number (int)`: Current player number, num==1 starts the game
        
        `timer: Timer`
            - a Timer object that can be used to fetch the remaining time for any player
            - Run `fetch_remaining_time(timer, player_number)` to fetch remaining time of a player
        """
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer
        self.total_time = fetch_remaining_time(self.timer, self.player_number)
        self.root = None
        self.me = 0
        self.enemy = 0

    def opening4(self, state: np.array) -> Tuple[int, int]:
        p = self.player_number
        e = 3 - p
        if self.enemy == 0:
            return (0, 0)
        elif self.enemy == 1 and self.me == 0:
            if state[0, 0] == e: return (1, 1)
            elif state[0, 3] == e: return (1, 3)
            elif state[0, 6] == e: return (1, 5)
            elif state[3, 0] == e: return (3, 1)
            elif state[3, 6] == e: return (3, 5)
            elif state[6, 3] == e: return (5, 3)
            elif state[1, 2] == e or state[2, 1] == e: return (0, 0)
            elif state[1, 4] == e or state[2, 5] == e: return (0, 6)
            elif state[4, 2] == e or state[4, 4] == e: return (6, 3)
            elif state[2, 1] == 0 and state[3, 0] == 0 and state[2, 0] == 0 and state[1, 0] == 0 and state[1, 1] == 0 and state[3, 1] == 0: return (2, 1)
            elif state[1, 2] == 0 and state[0, 1] == 0 and state[0, 2] == 0 and state[0, 3] == 0 and state[1, 1] == 0 and state[1, 3] == 0: return (1, 2)
        elif self.enemy == 1 and self.me == 1:
            if state[3, 0] == e or state[2, 0] == e or state[1, 0] == e or state[1, 1] == e or state[2, 1] == e or state[3, 1] == e: return (1, 2)
            else: return (2, 1)

        elif self.enemy == 2 and self.me == 1:
            if state[2, 1] == p and state[0, 3] == 0 and state[3, 0] == 0 and state[2, 0] == 0 and state[1, 0] == 0 and state[1, 1] == 0 and state[3, 1] == 0: return (3, 0)
            elif state[1, 2] == p and state[0, 1] == 0 and state[0, 2] == 0 and state[0, 3] == 0 and state[3, 0] == 0 and state[1, 1] == 0 and state[1, 3] == 0: return (0, 3)

        elif self.enemy == 2 and self.me == 2:
            if state[2, 1] == p and state[3, 0] == 0 and state[2, 0] == 0 and state[1, 0] == 0 and state[1, 1] == 0 and state[3, 1] == 0: return (3, 0)
            elif state[1, 2] == p and state[0, 1] == 0 and state[0, 2] == 0 and state[0, 3] == 0 and state[1, 1] == 0 and state[1, 3] == 0: return (0, 3)
        return None
    
    def opening6(self, state: np.array) -> Tuple[int, int]:
        p = self.player_number
        e = 3 - p
        if self.enemy == 0:
            return (5, 1)
        elif self.enemy == 1 and self.me == 0:
            if state[0, 5] == e or state[5, 10] == e: return (1, 5)
            elif state[0, 0] == e or state[9, 5] == e: return (5, 1)
            elif state[0, 10] == e: return (1, 5)
            elif state[5, 0] == e: return (1, 1)
        elif self.me == 1 and self.enemy == 1:
            if state[5, 0] == e: return (6, 1)
            elif state[6, 1] == 0 and state[6, 2] == 0: return (7, 2) 
            else: return (3, 0)
        return None
        
    def get_move(self, state: np.array) -> Tuple[int, int]:
        if self.me == 0 and self.enemy == 0:
            self.me = np.sum(state == self.player_number)
            self.enemy = np.sum(state == 3 - self.player_number)
        else:
            self.enemy += 1
        coverage = self.me + self.enemy
        # print(coverage, fetch_remaining_time(self.timer, self.player_number))
        dim = (state.shape[0] + 1) // 2
        
        # return self.get_move_mcts(state, 10)
        if dim == 4:
            if coverage < 5:
                opening = self.opening4(state)
                if opening is None: pass
                else: return tuple(opening)
            if coverage < 10:
                to_ret = self.get_move_mcts(state, 1000, C=0, k=50)
            elif coverage < 30:
                to_ret = self.get_move_mcts(state, 1000, C=0, k=50)
            else:
                to_ret = self.get_move_mcts(state, 1000, C=0, k=50)
            # if coverage < 75:
            #     return self.get_move_alpha_beta(state, 4)
            # else:
            #     return self.get_move_alpha_beta(state, 4)

        elif 4 < dim <= 6:
            if coverage < 3:
                opening = self.opening6(state)
                if opening is None: pass
                else: return tuple(opening)
            if coverage < 20:
                to_ret = self.get_move_mcts(state, 1750 , C=0, k=50)
            elif coverage < 35:
                to_ret = self.get_move_mcts(state, 1750, C=0, k=50)
            else:
                to_ret = self.get_move_mcts(state, 1750, C=0, k=50)
        
        else:
            if coverage < 10:
                to_ret = self.get_move_mcts(state, 1000 , C=0, k=50)
            elif coverage < 20:
                to_ret = self.get_move_mcts(state, 1000, C=0, k=50)
            else:
                to_ret = self.get_move_mcts(state, 1000, C=0, k=50)

        self.me += 1
        return tuple((int(to_ret[0]), int(to_ret[1])))

    def get_move_alpha_beta(self, state: np.array, depth=4) -> Tuple[int, int]:
        """
        Given the current state of the board, return the next move

        # Parameters
        `state: Tuple[np.array]`
            - a numpy array containing the state of the board using the following encoding:
            - the board maintains its same two dimensions
            - spaces that are unoccupied are marked as 0
            - spaces that are blocked are marked as 3
            - spaces that are occupied by player 1 have a 1 in them
            - spaces that are occupied by player 2 have a 2 in them

        # Returns
        Tuple[int, int]: action (coordinates of a board cell)
        """
        return self.iterative_deepening_alpha_beta(state, depth)[1]
    
    def iterative_deepening_alpha_beta(self, state: np.array, max_depth: int, budget=10):
        best_move = None
        move_order = []
        start_time = time.time()
        
        for depth in range(1, max_depth + 1):
            print(f"Starting depth-limited search at depth {depth}")
            alpha = -1
            beta = 1
            best_value = -np.inf
            valid_actions = get_valid_actions(state, self.player_number)
            
            if move_order:
                valid_actions = sorted(valid_actions, key=lambda x: move_order.index(x) if x in move_order else len(move_order))
            
            for move in valid_actions:
                value = self.alpha_beta_minimax(state, depth, 3 - self.player_number, move, alpha, beta)
                if value > best_value:
                    best_value = value
                    best_move = move
                if best_value >= beta: return best_value, best_move
                alpha = max(alpha, best_value)
                if time.time() - start_time > budget: return best_value, best_move
            
            move_order = [best_move] + [m for m in valid_actions if m != best_move]

        return best_value, best_move

    def _update_root_from_state(self, state: np.array):
        diffs = np.argwhere(self.root.state != state)
        move1, move2 = tuple(diffs[0]), tuple(diffs[1])

        if state[move1] == self.player_number:
            moves = [move1, move2]
        else:
            moves = [move2, move1]  
            
        children = [child for child in self.root.children if child.move == moves[0] and child.player == 3 - self.player_number]
        if len(children) == 0: return
        new_root = children[0]
        grandchildren = [child for child in new_root.children if child.move == moves[1] and child.player == self.player_number]
        
        if len(grandchildren) == 0: return
        self.root = grandchildren[0]
        self.root.parent = None
    
    def get_move_mcts(self, state: np.array, budget=5, C=2, k=50) -> Tuple[int, int]:
        dim = (state.shape[0] + 1) // 2
        valid_actions = get_valid_actions(state)
        dict_of_dsu = create_dict(state)

        if dim == 4:
            for move in valid_actions:
                if self.alpha_beta_minimax(state, 3, 3 - self.player_number, move, -2, 2) == 1: return move
            for move in valid_actions:
                new_state = state.copy()
                new_state[move[0], move[1]] = 3 - self.player_number
                if self.alpha_beta_minimax(state, 3, self.player_number, move, -2, 2) == -1: return move
        if dim > 4:
            for move in valid_actions:
                new_state = state.copy()
                new_state[move[0], move[1]] = self.player_number
                dsu_copy = deepcopy(dict_of_dsu)
                update_dsu(new_state, move, dsu_copy, vc=False)
                if check_win_own(new_state, dsu_copy, move): return move
            for move in valid_actions:
                new_state = state.copy()
                new_state[move[0], move[1]] = 3 - self.player_number
                dsu_copy = deepcopy(dict_of_dsu)
                update_dsu(new_state, move, dsu_copy, vc=False)
                if check_win_own(new_state, dsu_copy, move): return move

        if self.root:
            self._update_root_from_state(state)
            # print("updated root")

        if self.root is None or not np.array_equal(self.root.state, state):
            # print("manually updating root")
            self.root = MCTSNode(state, dict_of_dsu, player=self.player_number, actions=set(valid_actions), pnum=self.player_number)

        start_time = time.time()
        i = 0
        while i < budget:
            elap = time.time() - start_time
            if dim == 4 and elap > self.total_time / 20: break
            if dim > 4 and elap > self.total_time / 40: break
            node = self.root.tree_policy(weight=C, k=k)
            if node is None:
                node = self.root
            # print("tree+policy", time.time() - start_time)
            for _ in range(3):
                reward, rave = node.rollout()
                # print("reward", time.time() - start_time)
                node.backpropagate(reward, rave)
            # print("backpropagate", time.time() - start_time)
            i += 1

        # print(i)
        # print(self.root.children)
        best_move = self.root.best_child_rave(exploration_weight=0, k=k)
        if best_move is not None: 
            # print(best_move.dsu)
            best_move = best_move.move
        if best_move is None:
            # print('no child')
            # print(self.root)
            return random.choice(valid_actions)
        return best_move

    def alpha_beta_minimax(self, state: np.array, depth: int, maxPlayer: int, move: Tuple[int, int], alpha: int, beta: int) -> int:
        new_state = state.copy()
        new_state[move[0], move[1]] = 3 - maxPlayer
        
        if check_win(new_state, move, 3 - maxPlayer)[0]:
            if maxPlayer == 3 - self.player_number: 
                return 1
            else: 
                return -1
        elif depth == 0:
            new_state[move[0], move[1]] = 0
            value = self.heuristic(new_state, move, 3 - maxPlayer)
            return value
        
        valid_actions = get_valid_actions(new_state, self.player_number)
        fac = (2 * (maxPlayer == self.player_number) - 1)
        bestValue = -fac * math.inf
        
        for new_move in valid_actions:
            value = self.alpha_beta_minimax(new_state, depth - 1, 3 - maxPlayer, new_move, alpha, beta)
            bestValue = fac * max(fac * bestValue, fac * value)
            if maxPlayer == self.player_number:
                if bestValue >= beta:
                    break
                alpha = max(alpha, bestValue)
                
            else:
                if bestValue <= alpha:
                    break
                beta = min(beta, bestValue)
        
        return bestValue
    
    
    def heuristic(self, given_state: np.array, move: List[Tuple[int, int]], player: int) -> Tuple[int, int]:
        return 0
