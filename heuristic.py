import numpy as np

GAME_MARGIN_X = 16
GAME_HEIGHT = 144
GAME_WIDTH = 80

GRID_HEIGHT = 18
GRID_WIDTH = 10

W = 255
B = 0

DEBUG_MODE = False

# a simple tile grid, 255 if a tile is present, 0 if not
def get_grid(pixels):
    grid = np.mean(pixels, axis=2)
    grid = grid[0:GAME_HEIGHT:8, GAME_MARGIN_X:GAME_MARGIN_X+GAME_WIDTH:8]
    grid[grid > 200] = 255
    grid[grid <= 200] = 0
    grid = grid.astype('uint8')
    return grid

def explore(grid, x, y, result):
    if grid[y][x] == B and result[y][x] != B:
        result[y][x] = B
        if x + 1 < GRID_WIDTH:
            explore(grid, x + 1, y, result)
        if x - 1 >= 0:
            explore(grid, x - 1, y, result)
        if y - 1 >= 0:
            explore(grid, x, y - 1, result)

def remove_floating_tile(grid):
    result = np.zeros(grid.shape).astype('uint8')
    result.fill(W)
    for x in range(GRID_WIDTH):
        explore(grid, x, GRID_HEIGHT - 1, result)
    return result

# gets the maximum height of each tile column, ignores holes
def get_column_heights(grid):
    heights = []
    for x in range(GRID_WIDTH):
        height = 0
        for y in range(GRID_HEIGHT):
            tile = grid[GRID_HEIGHT - y - 1][x]
            if tile == B:
                height = y + 1
        heights.append(height)
    return np.array(heights)

def get_holes(grid, heights):
    total = 0
    for x in range(GRID_WIDTH):
        for y in range(heights[x] - 1):
            tile = grid[GRID_HEIGHT - y - 1][x]
            aboveTile = grid[GRID_HEIGHT - y - 2][x]
            if tile == W and aboveTile == B:
                total += 1
    return total

def get_complete_lines(grid):
    count = 0
    for r in range(GRID_HEIGHT):
        complete = True
        for c in range(GRID_WIDTH):
            if grid[r][c] == W:
                complete = False
        if complete:
            count += 1
    return count

def drop(grid, r, c, dist):
    grid[r][c] = W
    grid[r + dist][c] = B

def drop_tiles(grid):
    #grid = get_grid(pixels)
    noFloatGrid = remove_floating_tile(grid)
    diff = grid - noFloatGrid
    rawCoords = np.nonzero(diff)
    coords = []

    # get all tile coordinates
    for i in range(len(rawCoords[0])):
        coords.append((rawCoords[0][i], rawCoords[1][i]))

    # get the max distance to drop a tile
    heights = get_column_heights(noFloatGrid)
    maxDist = 1000
    for coordinate in coords:
        dist = GRID_HEIGHT - heights[coordinate[1]] - coordinate[0] - 1
        if dist < maxDist:
            maxDist = dist

    # drop the tiles
    for coordinate in coords:
        drop(grid, coordinate[0], coordinate[1], maxDist)


# heuristic value formula presented by:
# **** https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/
def estimate_value(pixels):
    grid = get_grid(pixels)
    drop_tiles(grid)
    heights = get_column_heights(grid)
    aggHeight = heights.sum()
    bumpiness = np.sum(np.abs(np.diff(heights)))
    holes = get_holes(grid, heights)
    completeLines = get_complete_lines(grid)

    reward = -0.51006 * aggHeight + 0.760666 * completeLines + -0.35663 * holes + -0.184483 * bumpiness

    if DEBUG_MODE:
        print('heights', heights)
        print('aggHeight', aggHeight)
        print('bumpiness', bumpiness)
        print('holes', holes)
        print('completeLines', completeLines)
        print('reward', reward)
        from PIL import Image
        Image.fromarray(grid).show()

    return reward


def main():
    from tetris_learning_environment.gym import TetrisEnvironment
    env = TetrisEnvironment('../Tetris.gb', frame_skip=60)
    env.reset()
    for _ in range(20):
        pixels = env.step(0)[0]
    estimate_value(pixels)

if __name__ == '__main__':
    main()
