import numpy as np
from PIL import Image

def reward(pixels):
    # now i have an nparray or pixels
    print(pixels.shape)
    img = pixels.astype('uint8')
    Image.fromarray(img).show()

def main():
    from tetris_learning_environment.gym import TetrisEnvironment
    env = TetrisEnvironment('../tetris.gb', frame_skip=60)
    env.reset()
    pixels, _, _ , _ = env.step(0)
    reward(pixels)

if __name__ == '__main__':
    main()