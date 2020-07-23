cdef tuple cy_playout1(int[:, :] board, int N):
    cell_size = int((size ** 2) / 2) + 10
    cdef int[:, :] black_rave = np.empty([cell_size, 2], dtype=np.int32)
    cdef int[:, :] white_rave = np.empty([cell_size, 2], dtype=np.int32)

    cdef int i, j, x, y, h
    i, j = 0, 0
    cdef int M,L
    M = board.shape[0]
    L = board.shape[1]
    for h in range(N):
        for x in range(M):
            for y in range(L):
                if board[x,y] == 0:
                    black_rave[i][0], black_rave[i][1] = x, y
                    i += 1
                elif board[x,y] == 1:
                    white_rave[j][0], white_rave[j][1] = x, y
                    j += 1
        i = 0
        j = 0

    return black_rave[:i], white_rave[:j]

