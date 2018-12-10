from cgp.genome import Genome

from multiprocessing import Pool
from timeit import default_timer as timer
import time
import copy

class TrainingEnvironment:
    OUTPUT_DIR = 'output/'

    def __init__(self):
        self.lastScore = 0

    def run(self, trainer, cgpConfig):
        elite = Genome(cgpConfig)
        bestScore = -1000000
        children = [Genome(cgpConfig) for _ in range(cgpConfig.childrenPerGeneration)]
        env = trainer.get_env()
        for generation in range(cgpConfig.generations):
            startTime = timer()

            elite.update_children(children)

            results = []
            for child in children:
                trainer.reset()
                results.append(trainer.run_episode(env, child))

            for (genome, score) in results:
                if score >= bestScore:
                    bestScore = score
                    genome.copy_into(elite)

            endTime = timer()

            self.log_generation(startTime, endTime, generation, cgpConfig.generations, bestScore)
            elite.save_to_file(self.OUTPUT_DIR + cgpConfig.modelFile)

        return (elite, bestScore)

    def run_parallel(self, trainer, cgpConfig, numProcesses):
        elite = Genome(cgpConfig)
        bestScore = -1000000

        children = [Genome(cgpConfig) for _ in range(cgpConfig.childrenPerGeneration)]
        trainers = [copy.deepcopy(trainer)] * len(children)
        envs = [trainer.get_env()] * len(children)


        for generation in range(cgpConfig.generations):
            startTime = timer()

            elite.update_children(children)

            pool = Pool(processes=numProcesses, initializer=self.init_pool, initargs=(), maxtasksperchild=1)
            asyncHandles = self.run_children_async(pool, trainers, envs, children)
            pool.close()
            pool.join()
            results = self.wait(asyncHandles)
            for (genome, score) in results:
                if score >= bestScore:
                    bestScore = score
                    genome.copy_into(elite)


            endTime = timer()
            elite.save_to_file(cgpConfig.modelFile)
            self.log_generation(startTime, endTime, generation, cgpConfig.generations, bestScore)

        return (elite, bestScore)

    def init_pool(self):
        pass

    def get_children(self, genome, count):
        return [genome.get_child() for _ in range(count)]

    def run_children_async(self, pool, trainers, envs, children):
        results = []
        for i, _ in enumerate(children):
            trainers[i].reset()
            results.append(pool.apply_async(trainers[i].run_episode, args=(envs[i], children[i])))
        return results

    def wait(self, asyncHandles):
        return [handle.get() for handle in asyncHandles]

    def log_generation(self, startTime, endTime, generation, totalGenerations, bestScore):
        timeElapsed = endTime - startTime
        estimatedTimeSec = timeElapsed * (totalGenerations + 1 - generation)
        estimatedTimeMin = estimatedTimeSec / 60.0
        print('Generation ' + str(generation + 1) + ' of ' + str(totalGenerations) + ' complete, current best score = ', bestScore)
        print('Est. minutes remaining: ' + str(estimatedTimeMin))
