import numpy as np
import networkx as nx
import random
import copy
from math import ceil, floor, exp, sqrt, log
from time import time
import gc
import json
from matplotlib import pyplot as plt
import code

'''================================================================================================================='''
'''===================================================GAME code====================================================='''
'''================================================================================================================='''

hashTable = {}
hashTurn = random.getrandbits(64)
for state in range(-1, 2): #Blue, Red or Empty
    for cell in range(61): #board_size (number of cells)
        hashTable[(cell, state)] = random.getrandbits(64)

class HexBoard:
    def __init__(self, 
            radius=4, 
            mode="TRIAGO",
        ) -> None:
        self.radius = radius
        self.mode = mode

        self.players_to_id = {"Blue" : -1, "Red" : 1}
        self.id_to_players = {-1 : "Blue", 1 : "Red"}
        self.steps = np.array([[-1,1], [-1,0], [0,-1], [1,-1], [1, 0], [0, 1]])
        self.turn = 1
        self.init_board()
        self.init_board_edges()
        self.comp_graph = {
            -1 : self.create_complementary_graph(-1),
            1 : self.create_complementary_graph(1),
        }
        # Transposition table
        self.board_size = (lambda n:3*(n**2) - 3*n + 1)(radius+1)
        self.tuple_to_board_idx = {node:i for i, node in enumerate(self.graph.nodes.keys()) if not isinstance(node, str)}
        self.board_idx_to_tuple = {v:k for k,v in self.tuple_to_board_idx.items()}
        self.h = 0

    def place_piece(self, q, r, player=None):
        '''pose un pion sur la cellule (q,r) de la couleur player'''
        if isinstance(player, str):
            player = self.players_to_id[player]
        if player is None:
            player = self.turn

        # Check if the cell can be linked to two different board edges
        if not(self.is_valid_move(q, r, player)):
            return False
        self.graph.nodes[(q, r)]['player'] = player

        # met à jour le graphe, en ajoutant les liens qui ont potentiellement été crées
        for neighbor in self.comp_graph[player].neighbors((q, r)):
            if self.graph.nodes[neighbor]["player"] == player:
                self.graph.add_edge((q, r), neighbor)

        # met à jour le graphe complémentaire, en enlevant la cellule jouée pour l'adversaire
        self.comp_graph[-player].remove_node((q, r))

        #Transposition table
        global Table
        pos = self.tuple_to_board_idx.get((q,r))
        self.h = self.h ^ hashTable[(pos, self.turn)]
        # self.h = self.h ^ hashTable[(pos, 0)]
        self.h = self.h ^ hashTurn
        if self.h not in Table:
            self.add_to_table()

        # next turn
        self.turn *= -1

    def lookup(self):
        global Table
        return Table.get(self.h, None)
        # return Table.get(self.current_hash, None)
    
    def add_to_table(self):
        global Table
        nplayouts = np.zeros(self.board_size)
        nwins = np.zeros(self.board_size)
        rave_visits = np.zeros(self.board_size)
        rave_wins = np.zeros(self.board_size)
        Table[self.h] = [0, nplayouts, nwins, rave_visits, rave_wins]
        # Table[self.current_hash] = [0, nplayouts, nwins, rave_visits, rave_wins]

    def init_board(self):
        self.graph = nx.Graph()
        for q in range(-self.radius, self.radius + 1):
            for r in range(-self.radius, self.radius + 1):
                if -q - r >= -self.radius and -q - r <= self.radius:
                    self.graph.add_node((q, r))
                    self.graph.nodes[(q, r)]['player'] = None

    def init_board_edges(self):
        '''Creates super nodes representing the board edges'''
        player = 1
        if self.mode == "TRIAGO":
            for i in range(6):
                self.graph.add_node(f"Board_edge{i}")
                self.graph.nodes[f"Board_edge{i}"]["player"] = player
                player *= -1
        self.board_edges = set(node for node in self.graph.nodes if isinstance(node, str))
    
    def create_complementary_graph(self, player:int) -> nx.Graph:
        # But: créer un graphe avec seulement les cellules vides, et les cellules du joueur 
        # Equivalent à un graphe des cellules n'appartenant pas à l'adversaire
        nodes = self.graph.nodes
        complementary_graph = nx.Graph()
        for node in nodes:
            if isinstance(node, str):
                continue
            if nodes[node]['player'] != -player:
                complementary_graph.add_node(node)
                # relier les cellules adjacentes
                for adjacent_cell in self.get_surrounding_cells(*node):
                    if adjacent_cell in complementary_graph.nodes:
                        complementary_graph.add_edge(adjacent_cell, node)

        # ajout des ports
        pos = np.array([self.radius, 0])
        for i, step in enumerate(self.steps):
            if tuple(pos) in complementary_graph.nodes and self.graph.nodes[f"Board_edge{i}"]['player'] == player:
                complementary_graph.add_edge(tuple(pos), f"Board_edge{i}")
            for _ in range(self.radius):
                pos += step
                if tuple(pos) in complementary_graph.nodes and self.graph.nodes[f"Board_edge{i}"]['player'] == player:
                    complementary_graph.add_edge(tuple(pos), f"Board_edge{i}")
        return complementary_graph
        
    def get_surrounding_cells(self, q, r):
        return [(q + dq, r + dr) for (dq, dr) in ((-1,1), (-1,0), (0,-1), (1,-1), (1,0), (0,1))
                if (q + dq, r + dr) in self.graph.nodes]
            
    def is_valid_move(self, q, r, player):
        """Check if placing a piece at (q, r) for the given player is a valid move."""
        # Make sure the position is on the board
        if (q, r) not in self.graph.nodes:
            return False

        # Verify this cell is available (this excludes board edges and cells of the ennemy)
        if self.graph.nodes[(q, r)]['player'] is not None:
            return False
        
        # Run DFS to check if (q, r) can be connected to a board edge cell of the same color
        self.visited = set(nx.dfs_tree(self.comp_graph[player], source=(q, r)))
        return (len(self.visited.intersection(self.board_edges)) >= 2)
    
    def get_captured_cells(self, player:int) -> set:
        return {cell for cell, attribute in self.graph.nodes.items() if attribute['player'] == player}
    
    def legalMoves(self):
        empty_cells = self.get_captured_cells(None)
        empty_cells_ = empty_cells.copy()
        legals = set()
        for empty_cell in empty_cells_:
            if self.is_valid_move(*empty_cell, self.turn):
                legals.update((empty_cell,))
                # when checking if empty_cell was a valid move, we used DFS and stored the visited cells, we can 
                # use this to automatically asign these visited cells as valid moves
                already_visited = self.visited.difference(self.board_edges).intersection(empty_cells_)
                legals.update(already_visited)
                empty_cells.difference_update(already_visited)
            else:
                empty_cells.remove(empty_cell)
            if len(empty_cells) == 0:
                break
        return list(legals)
    
    def is_terminal(self) -> bool:
        return len(self.legalMoves()) == 0
    
    def playout(self) -> float:
        self.played = []
        while not self.is_terminal():
            moves = self.legalMoves()
            random_move = moves[random.randint(0, len(moves) - 1)]
            self.place_piece(*random_move)
            # self.visited_positions.append((self.current_hash, random_move))
            self.played.append(self.tuple_to_board_idx.get(random_move))
        draw = (len(self.get_captured_cells(-1)) + len(self.get_captured_cells(1)) == self.board_size+6)
        result = 0.5 if draw else float(self.turn == 1) # blue's score
        return result

    def restart_game(self) -> None:
        self.turn = 1
        self.init_board()
        self.init_board_edges()
        self.comp_graph = {
            -1 : self.create_complementary_graph(-1),
            1 : self.create_complementary_graph(1),
        }

    def check_state(self) -> dict:
        return dict(self.graph.nodes.items())

'''================================================================================================================='''
'''===================================================Algorithm====================================================='''
'''================================================================================================================='''

class Flat_MC:

    def __init__(self, n_playouts:int):
        self.n_playouts = n_playouts

    def search(self, board:HexBoard):
        moves = board.legalMoves()
        bestScore = 0
        bestMove = 0
        playouts_per_move = self.n_playouts // len(moves)
        # For every move, try the move then play it out n // n_moves times with random moves. Pick the move who had the highest outcome
        for m in range(len(moves)):
            cum_score = 0
            for i in range (playouts_per_move):
                b = copy.deepcopy(board)
                b.place_piece(*moves[m])
                r = b.playout() # Blue player's score: 1 if he won, 0 if he lost
                cum_score += r # Blue player's cumulative score
            # At the end of the inner loop, if the player is Red, we can compute his score by doing i - cum_score. 
            # In other words, Red won all the playout games Blue didn't win
            if board.turn == 1:
                cum_score = playouts_per_move - cum_score
            if cum_score > bestScore:
                bestScore = cum_score
                bestMove = m
        return moves[bestMove]
    
class UCB:

    def __init__(self, n_playouts:int, c:float):
        self.n_playouts = n_playouts
        self.c = c

    def search(self, board:HexBoard) -> tuple:
        moves = board.legalMoves()
        n_moves = len(moves)
        sumScores = np.zeros(n_moves)
        nbVisits = np.zeros(n_moves)
        
        # First: play each move once. This ensures we don't divide by 0 in the formula
        for m in range(n_moves):
            b = copy.deepcopy(board)
            b.place_piece(*moves[m])
            r = b.playout()
            if board.turn == 1: #If initial turn was red's, red's score is 1 - blue's score
                r = 1 - r
            sumScores[m] += r
            nbVisits[m] += 1
        # Then use the UCB formula to select the move
        for t in range(n_moves+1, self.n_playouts+1):
            # we can compute the UCB scores every move using numpy's array operations
            UCB_scores = (sumScores / nbVisits) + self.c * np.sqrt(log(t) / nbVisits)
            best_move = np.argmax(UCB_scores)

            b = copy.deepcopy(board)
            b.place_piece(*moves[best_move])
            r = b.playout()
            if board.turn == 1:
                r = 1 - r
            sumScores[best_move] += r
            nbVisits[best_move] += 1

        return moves[np.argmax(nbVisits)]
    
class UCT:

    def __init__(self, n_playouts:int, c:float):
        self.n_playouts = n_playouts
        self.c = c

    def UCT_playout(self, board:HexBoard):
        if board.is_terminal():
            return board.turn == 1 #Blue's score
        t = board.lookup()
        if t is not None :
            bestValue = -np.inf
            moves = board.legalMoves()
            for m in range(len(moves)) :
                n = t[0]
                ni = t[1][m]
                wi = t[2][m]
                if ni > 0 :
                    Q = wi/ni
                    if board.turn == 1:
                        Q = 1 - Q
                    value = Q + self.c * sqrt(log(n)/ni)
                else:
                    value = np.inf
                if value > bestValue :
                    bestValue = value
                    bestMove = m
            board.place_piece(*moves[bestMove])
            res = self.UCT_playout(board)
            t[0] += 1
            t[1][bestMove] += 1
            t[2][bestMove] += res
            return res
        else :
            board.add_to_table()
            return board.playout()
        
    def search(self, board:HexBoard):
        global Table
        Table = dict()
        for i in range(self.n_playouts):
            b = copy.deepcopy(board)
            res = self.UCT_playout(b)
        t = board.lookup()
        moves = board.legalMoves()
        # return the most chosen move by the UCT formula
        # code.interact(local=locals())
        return moves[np.argmax(t[1])]

class RAVE:

    def __init__(self, n_playouts:int, c:float, rave_beta:float):
        self.n_playouts = n_playouts
        self.c = c
        self.rave_beta = rave_beta

    def UCT_RAVE(self, board: HexBoard):
        if board.is_terminal():
            return board.turn == 1  # Blue's score
        t = board.lookup()
        if t is not None:
            bestValue = -np.inf
            moves = board.legalMoves()
            for m in range(len(moves)):
                value = 100000000
                n = t[0]
                ni = t[1][m]
                wi = t[2][m]
                nri = t[3][m]
                wri = t[4][m]
                if ni > 0:
                    Q = wi / ni
                    if board.turn == 1:
                        Q = 1 - Q
                    Q_rave = wri / nri if nri > 0 else 0
                    beta = self.rave_beta / (n + self.rave_beta)
                    value = (1 - beta) * Q + beta * Q_rave + self.c * sqrt(log(n) / ni)
                if value > bestValue:
                    bestValue = value
                    bestMove = m
            board.place_piece(*moves[bestMove])
            res = self.UCT_RAVE(board)
            t[0] += 1
            t[1][bestMove] += 1
            t[2][bestMove] += res
            t[3][bestMove] += 1
            t[4][bestMove] += res
            return res
        else:
            board.add_to_table()
            t = board.lookup()
            result = board.playout()
            self.update_amaf(t, board.played, result)
            return result
        
    def update_amaf(self, t:list, played:list, result:float):
        for i in range(len(played)):
            if played[:i].count(played[i]) == 0 :
                t[3][played[i]] += 1
                t[4][played[i]] += result

    def search(self, board:HexBoard):
        global Table
        Table = dict()
        for i in range(self.n_playouts):
            b = copy.deepcopy(board)
            self.UCT_RAVE(b)
        t = board.lookup()
        moves = board.legalMoves()
        return moves[np.argmax(t[1])]

class SH:
    '''Sequential Halving'''
    def __init__(self, n_playouts:int, c:float):
        self.n_playouts = n_playouts
        self.c = c

    def UCT_playout(self, board:HexBoard):
        if board.is_terminal():
            return board.turn == 1 #Blue's score
        t = board.lookup()
        if t is not None :
            bestValue = -np.inf
            moves = board.legalMoves()
            for m in range(len(moves)) :
                n = t[0]
                ni = t[1][m]
                wi = t[2][m]
                if ni > 0 :
                    Q = wi/ni
                    if board.turn == 1:
                        Q = 1 - Q
                    value = Q + self.c * sqrt(log(n)/ni)
                else:
                    value = np.inf
                if value > bestValue :
                    bestValue = value
                    bestMove = m
            board.place_piece(*moves[bestMove])
            res = self.UCT_playout(board)
            t[0] += 1
            t[1][bestMove] += 1
            t[2][bestMove] += res
            return res
        else :
            board.add_to_table()
            return board.playout()

    def search(self, board:HexBoard):
        global Table
        Table = dict()
        board.add_to_table()
        #utils
        idx_to_move = {board.tuple_to_board_idx[move]:move for move in board.legalMoves()}
        n_moves = len(idx_to_move)
        board_idx = set(board.tuple_to_board_idx.values())

        n_playouts_per_move = floor(self.n_playouts//(n_moves*np.log2(n_moves))) if n_moves > 1 else self.n_playouts
        playouts = np.zeros(board.board_size) + 1e-8
        wins = np.zeros(board.board_size)
        while (len(idx_to_move)>1):
            not_used_moves = np.array(list(board_idx - set(idx_to_move.keys())))
            for idx, move in idx_to_move.items():
                for _ in range(n_playouts_per_move):
                    b = copy.deepcopy(board)
                    b.place_piece(*move)
                    res = self.UCT_playout(b)
                    playouts[idx] += 1
                    wins[idx] += res if board.turn == -1 else 1 - res
            if len(not_used_moves) > 0:
                # this ensures we don't divide by 0
                playouts[not_used_moves] += 1.0
                # turn-off not used moves
                wins[not_used_moves] = -1.0
            idx_to_move = self.BestHalf(idx_to_move, wins, playouts)
        return list(idx_to_move.values())[0]

    def BestHalf(self, idx_to_move:dict, wins:np.ndarray, playouts:np.ndarray):
        middle = ceil(len(idx_to_move)/2)
        win_rate = wins / playouts
        # np.partition(array, k) does k iterations of the sorting loop, it guarantees that the kth smallest (or largest) elements are
        # sorted. np.argpartition returns the index instead of values. 
        # np.argpartition(arr, -k)[-k:] gives the indexes of the k largest elements (like torch.topk().indices)
        sorted_indexes = np.argpartition(win_rate, -middle)
        idx_to_move =  {largest_index:idx_to_move[largest_index] for largest_index in sorted_indexes[-middle:]}
        return idx_to_move


class Tournament:

    def __init__(
            self, 
            algorithms:dict[str : Flat_MC|UCT|UCB|RAVE|SH],
            game_per_pair=10,
            starting_elos=None,
            radius=4,
            k=32,
        ) -> None:
        self.algorithms = algorithms
        self.games_per_pair = game_per_pair
        self.radius = radius
        self.k = k
        if starting_elos is not None:
            with open("elo.json", "r") as f:
                self.elo_ratings = json.load(f)
        else:
            self.elo_ratings = {
                alg_name:1000 for alg_name in algorithms.keys() 
            }
        self.history = []

    def calculate_elo(self, elo_p1, elo_p2, score):
        expected_score = 1 / (1 + 10 ** ((elo_p2 - elo_p1) / 400))
        new_rating = elo_p1 + self.k * (score - expected_score)
        return new_rating

    def add_player(self, name, algorithm, elo=1000):
        self.algorithms[name] = algorithm
        self.elo_ratings[name:elo]

    def organize_matchups(self):
        '''play game_per_pair games between each possible pair in a random order'''
        self.matchups = []
        for i, name_i in enumerate(self.algorithms.keys()):
            for j, name_j in enumerate(self.algorithms.keys()):
                if i<=j:
                    continue
                for _ in range(self.games_per_pair):
                    # randomly set the first player to play
                    if random.randint(0,1):
                        self.matchups.append((name_i, name_j))
                    else:
                        self.matchups.append((name_j, name_i))
        random.shuffle(self.matchups)

    def play_game(self, matchup:tuple[str, str]) -> tuple:
        p1, p2 = matchup
        algorithms = {
            1:self.algorithms[p1],
            -1:self.algorithms[p2],
        }
        
        board_ = HexBoard(radius=self.radius, mode="TRIAGO")
        i = 0
        print(f"GAME: {p1} vs {p2} | {p1} starts")
        while not(board_.is_terminal()):
            i += 1
            t = time(); t_ = t
            print(f"Iteration {i} | {board_.id_to_players[board_.turn]}'s turn:")
            move = algorithms[board_.turn].search(board_)
            board_.place_piece(*move)
            t = time()
            print(f"Played {move} | Time: {t - t_:.4f}s")
            if i % 10 == 0: 
                gc.collect()

        print("="*30)
        draw = (len(board_.get_captured_cells(-1)) + len(board_.get_captured_cells(1)) == board_.board_size+6)
        result = 0.5 if draw else float(board_.turn == -1) # p1's score
        winner = p1 if board_.turn == -1 else p2
        print("RESULT:", winner if not draw else "draw")
        return result

    def play_all_matchups(self):
        for match_up in self.matchups:
            result = self.play_game(match_up)
            self.update_ratings(*match_up, result)
        with open('elo.json', 'w') as f:
            json.dump(self.elo_ratings, f)

    def update_ratings(self, alg1_name, alg2_name, result):
        '''
        result: the score of alg1
        '''
        alg1_rating = self.elo_ratings[alg1_name]
        alg2_rating = self.elo_ratings[alg2_name]

        if result == 0.5:
            alg1_score, alg2_score = 0.5, 0.5
        else:
            alg1_score, alg2_score = result, 1-result

        new_alg1_rating = self.calculate_elo(alg1_rating, alg2_rating, alg1_score)
        new_alg2_rating = self.calculate_elo(alg2_rating, alg1_rating, alg2_score)
        self.elo_ratings[alg1_name] = new_alg1_rating
        self.elo_ratings[alg2_name] = new_alg2_rating
        self.history.append(self.elo_ratings.copy())

    def plot_elo_history(self):
        """
        Plot the history of Elo ratings.
        """
        plt.figure(figsize=(12, 8))
        for alg_name in self.algorithms.keys():
            ratings = [history[alg_name] for history in self.history]
            plt.plot(ratings, label=alg_name)
        plt.xlabel('Game')
        plt.ylabel('Elo Rating')
        plt.title('Elo Rating Over Time')
        plt.legend()
        plt.show()

def main():
    global Table
    Table = dict() # this is just to make sure the HexBoard class works for algortihms that don't use the transposition table
    algorithms = {
        "FLAT_p100":Flat_MC(n_playouts=100),
        "UCB_p100_c0.4":UCB(n_playouts=100, c=0.4),
        "UCT_p100_c0.4":UCT(n_playouts=100, c=0.4),
        "RAVE_p100_c0.4_b0.1":RAVE(n_playouts=100, c=0.4, rave_beta=0.1),
        "RAVE_p100_c0.4_b1.0":RAVE(n_playouts=100, c=0.4, rave_beta=1.0),
        "RAVE_p100_c0.4_b10.0":RAVE(n_playouts=100, c=0.4, rave_beta=10.0),
        "SHOT_p500_c0.4":SH(n_playouts=500, c=0.4)
    }
    #!!! 10 games_per_pair ~~ 8 hours run CPU !!!
    tournoi = Tournament(algorithms, game_per_pair=1)
    tournoi.organize_matchups()
    print(tournoi.matchups)
    tournoi.play_all_matchups()
    tournoi.plot_elo_history()
    # opens the terminal with python and its local variables 
    code.interact(local=locals())


if __name__ == "__main__":
    main()
