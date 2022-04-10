# Function to solve the board
def solve(board):
    find = find_empty(board)
    if not find:
        return True
    else:
        row,col = find
    for i in range(1,10):
        if valid(board,i,(row,col)):
            board[row][col] = i
            if solve(board):
                return True
            board[row][col] = 0
    return False

# Function to find empty cell in the board
def find_empty(board):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return (i,j)
    return None

# Function to find check if the moveis valid
def valid(board,num,pos):
    for i in range(len(board[0])):
        if board[pos[0]][i] == num and pos[1] != i:
            return False

    for i in range (len(board)):
        if board[i][pos[1]] == num and pos[0] != i:
            return False

    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if board[i][j] == num and (i,j) != pos:
                return False

    return True