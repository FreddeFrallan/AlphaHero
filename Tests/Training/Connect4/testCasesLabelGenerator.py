from Main.MCTS.SequentialMCTS import Node

# ****** Simple one step Tree
ROOT = Node('A')
ROOT.expanded = True
ROOT.score = 0
childScores = [5, 6, 8, 10, 8, 6, 6]
childNames = ['B', 'C', 'D', 'E', 'F', 'G', 'H']
for a in range(len(childNames)):
    newChild = Node(childNames[a], a)
    newChild.visits = 10
    newChild.score = childScores[a]
    newChild.terminal = a > 3  # Just for random, should not affect anything

    ROOT.children.append(newChild)
    ROOT.visits += 10
    ROOT.score += childScores[a]

CASES = [
    [ROOT,
     ['A'],  # StateInput
     [[0.7]],  # Value Label
     [[0.10204082, 0.12244898, 0.16326531, 0.20408163, 0.16326531, 0.12244898, 0.12244898]]  # Policy Label
     ]
]
