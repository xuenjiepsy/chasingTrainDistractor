import numpy as np
import pandas as pd

class ComputeStatistics:
    def __init__(self, getTrajectories, measurementFunction):
        self.getTrajectories = getTrajectories
        self.measurementFunction = measurementFunction

    def __call__(self, oneConditionDf):
        allTrajectories = self.getTrajectories(oneConditionDf)
        allMeasurements = np.array([self.measurementFunction(trajectory) for trajectory in allTrajectories])
        # print(allMeasurements)
        measurementMean = np.mean(allMeasurements, axis = 0)
        measurementStd = np.std(allMeasurements, axis = 0)

        return pd.Series({'mean': measurementMean, 'std': measurementStd})

class SampleTrajectory:
    def __init__(self, maxRunningSteps, transit, isTerminal, rewardFunc, reset):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.rewardFunc = rewardFunc
        self.reset = reset

    def __call__(self, policy):
        # epsReward = np.array([0, 0, 0])
        state = self.reset()
        while self.isTerminal(state):
            print('reset')
            state = self.reset()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                # print('terminal------------')
                break
            action = policy(state)
            # print(action)
            nextState = self.transit(state, action)

            reward = self.rewardFunc(state, action, nextState)
            # print('state: ', state, 'action: ', action, 'nextState: ', nextState, 'reward: ', reward)
            # epsReward += reward

            trajectory.append((state, action, reward, nextState))

            state = nextState
        # print('epsReward: ', epsReward)
        return trajectory

class SampleExpTrajectory:
    def __init__(self, maxRunningSteps, transit, isTerminal, rewardFunc, reset):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.rewardFunc = rewardFunc
        self.reset = reset

    def __call__(self, policy):
        # epsReward = np.array([0, 0, 0])
        state = self.reset()
        while self.isTerminal(state):
            print('reset')
            state = self.reset()

        trajectory = []
        expTrajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                # print('terminal------------')
                break
            action = policy(state)
            # print(action)
            nextState = self.transit(state, action)

            reward = self.rewardFunc(state, action, nextState)
            # print('state: ', state, 'action: ', action, 'nextState: ', nextState, 'reward: ', reward)
            # epsReward += reward
            expTrajectory.append([[agentState[0],agentState[1]] for agentState in state])

            trajectory.append((state, action, reward, nextState))

            state = nextState
        expTrajectory.append([[agentState[0], agentState[1]] for agentState in state])
        # print([[agentState[0], agentState[1]] for agentState in state])
        # print('epsReward: ', epsReward)
        return trajectory, expTrajectory

class SampleExpTrajectoryWithAllFrames:
    def __init__(self, maxRunningSteps, transit, isTerminal, reset):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset

    def __call__(self, policyForWolfAndSheep, policyForMaster, policyForDistractor, observeForWolfAndSheep, observeForMaster, observeForDistractor):
        # epsReward = np.array([0, 0, 0])
        state = self.reset()
        while self.isTerminal(state):
            print('reset')
            state = self.reset()

        trajectory = []
        expTrajectory = []
        expTrajectory.append([[agentState[0],agentState[1]] for agentState in state])
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                # print('terminal------------')
                break
            stateForWolfAndSheep = observeForWolfAndSheep(state)
            stateForMaster = observeForMaster(state)
            stateForDistractor = observeForDistractor(state)
            actionForWolfAndSheep = policyForWolfAndSheep(stateForWolfAndSheep)
            actionForMaster = policyForMaster(stateForMaster)
            actionForDistractor = policyForDistractor(stateForDistractor)
            action = actionForWolfAndSheep + actionForMaster + actionForDistractor
            # print(action)
            nextState,nextAllStates = self.transit(state, action)

            # print('state: ', state, 'action: ', action, 'nextState: ', nextState, 'reward: ', reward)
            # epsReward += reward
            [expTrajectory.append([[agentState[0],agentState[1]] for agentState in frameState]) for frameState in nextAllStates]

            trajectory.append((state, action, nextState))

            state = nextState
        # expTrajectory.append([[agentState[0], agentState[1]] for agentState in state])
        # print([[agentState[0], agentState[1]] for agentState in state])
        # print('epsReward: ', epsReward)
        return trajectory, expTrajectory

class SampleExpTrajectoryWithAllFramesRecreate:
    def __init__(self, maxRunningSteps, transit, isTerminal, reset):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset

    def __call__(self, policyForWolfAndSheep, policyForMaster, policyForDistractor, observeForWolfAndSheep, observeForMaster, observeForDistractor):
        # epsReward = np.array([0, 0, 0])
        state = self.reset()
        while self.isTerminal(state):
            print('reset')
            state = self.reset()
        # state[0:3] = demoTraj[18][0][0][0:3]
        # state[1] = demoTraj[18][0][0][1]
        # state[2] = demoTraj[18][0][0][2]
        trajectory = []
        expTrajectory = []
        expTrajectory.append([[agentState[0],agentState[1]] for agentState in state])
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                # print('terminal------------')
                break
            stateForWolfAndSheep = observeForWolfAndSheep(state)
            stateForMaster = observeForMaster(state)
            stateForDistractor = observeForDistractor(state)
            actionForWolfAndSheep = policyForWolfAndSheep(stateForWolfAndSheep)
            actionForMaster = policyForMaster(stateForMaster, state)
            actionForDistractor= policyForDistractor(stateForDistractor)
            action = actionForWolfAndSheep + actionForMaster + actionForDistractor
            # print(action)
            nextState,nextAllStates = self.transit(state, action)
            # print('44444444444444444444444444')
            # print(nextState[2])
            # nextState[0:3] = demoTraj[18][runningStep][2][0:3]
            # nextAllStates[runningStep] = nextState




            # print('state: ', state, 'action: ', action, 'nextState: ', nextState, 'reward: ', reward)
            # epsReward += reward
            # expTrajectory.append([[agentState[0],agentState[1]] for agentState in nextState])
            [expTrajectory.append([[agentState[0],agentState[1]] for agentState in frameState]) for frameState in nextAllStates]
            # [trajectory.append(frameState) for frameState in nextAllStates]

            trajectory.append((state, action, nextState))

            state = nextState
        # expTrajectory.append([[agentState[0], agentState[1]] for agentState in state])
        # print([[agentState[0], agentState[1]] for agentState in state])
        # print('epsReward: ', epsReward)
        return trajectory, expTrajectory