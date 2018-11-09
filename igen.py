
from PIL import Image
from gene import Gene

import config
import random
import numpy as np

class ImageGenerator():
    IMAGE_WIDTH = 32  # must be a 255x255 image for now
    IMAGE_HEIGHT = 32

    INPUTS = 2
    OUTPUTS = 3

    def main(self):
        genome = self.initGenome()
        pixels = self.decodeGenome(genome)
        self.showImage(pixels)

    def imageFromRow(self, row):
        return row.reshape(3, 32, 32).transpose([1, 2, 0])

    def get_random_doggo(self):
        return self.imageFromRow(self.dogs[random.randint(0, len(self.dogs) - 1)])

    def doggos(self, dogImages):
        self.dogs = dogImages
        outputs = self.initOutputs()
        genome = self.initGenome()
        pixels = self.mutate(genome, outputs)
        self.showImage(pixels)

    def mutate(self, genome, outputs):
        total = 10000
        dog = self.get_random_doggo()
        self.showImage(dog)
        min = self.measure(self.decodeGenome(genome, outputs), dog)
        for i in xrange(total):
            print 'Progress...', str(i + 1), '/', str(total)

            child = list(genome)
            self.mutate_child(child)
            nextOutputs = self.initOutputs()
            childPixels = self.decodeGenome(child, nextOutputs)
            
            measure = self.measure(childPixels, dog)

            if measure < min:
                outputs = nextOutputs
                genome = child
                min = measure
                print 'New MSE: ', measure
                self.showImage(childPixels)

        return self.decodeGenome(genome, outputs)
            

    def measure(self, pixels, dog):
        return np.square(pixels - dog).mean()
    
    def mutate_child(self, child):
        for gene in child:
            i = random.randint(1, 3)
            if i == 1:
                index = random.randint(0, len(child) - 1)
                gene.mutate()
    
    def initOutputs(self):
        outputs = []
        for i in xrange(self.OUTPUTS):
            outputs.append(random.randint(0, config.GENOME_SIZE))
        return outputs

    def initGenome(self):
        genome = []
        for i in xrange(self.INPUTS):
            genome.append(Gene(i))
        for i in xrange(config.GENOME_SIZE):
            genome.append(Gene(self.INPUTS + i))
        return genome

    def decodeGenome(self, genome, outputs):
        pixels = np.zeros(
            (self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 3), dtype=np.uint8)
        for y in xrange(self.IMAGE_WIDTH):
            for x in xrange(self.IMAGE_HEIGHT):
                r = genome[outputs[0]].compute(genome, x, y)
                g = genome[outputs[1]].compute(genome, x, y)
                b = genome[outputs[2]].compute(genome, x, y)
                pixels[x][y] = [r, g, b]
        return pixels

    def showImage(self, pixels):
        Image.fromarray(pixels, mode='RGB').show()
