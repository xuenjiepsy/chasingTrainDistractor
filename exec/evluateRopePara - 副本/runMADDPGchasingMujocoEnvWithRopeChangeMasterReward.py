import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numpy as np
import json
import xmltodict
import mujoco_py as mujoco
import math

from src.maddpg.trainer.myMADDPG import BuildMADDPGModels, TrainCritic, TrainActor, TrainCriticBySASR, \
    TrainActorFromSA, TrainMADDPGModelsWithBuffer, ActOneStep, actByPolicyTrainNoisy, actByPolicyTargetNoisyForNextState, ActOneStepRecreate
from src.RLframework.RLrun_MultiAgent import UpdateParameters, SampleOneStep, SampleFromMemory,\
    RunTimeStep, RunEpisode, RunAlgorithm, getBuffer, SaveModel, StartLearn
from src.functionTools.loadSaveModel import saveVariables
from env.multiAgentMujocoEnv import RewardSheep, RewardWolf,RewardMaster, Observe, IsCollision, getPosFromAgentState,getVelFromAgentState,PunishForOutOfBound,PunishForOutOfBoundVarRange,ReshapeAction, TransitionFunctionWithoutXPos, ResetUniformWithoutXPosForLeashed, BoundForWolves,BoundForWolfself
from src.functionTools.editEnvXml import transferNumberListToStr,MakePropertyList,changeJointProperty
from src.functionTools.loadSaveModel import saveToPickle, restoreVariables,GetSavePath

# fixed training parameters
maxEpisode = 90000#150000
learningRateActor = 0.01#
learningRateCritic = 0.01#
gamma = 0.95 #
tau=0.01 #
bufferSize = 1e6#
minibatchSize = 1024#


# arguments: numWolves numSheeps numMasters saveAllmodels = True or False

def main():
    debug = 0
    if debug:

        damping=0.5
        frictionloss=0.1
        masterForce=1.0


        numWolves = 1
        numSheeps = 1
        numMasters = 1
        numDistractor = 2
        maxTimeStep = 25
        visualize=False
        saveAllmodels = True

    else:
        print(sys.argv)
        condition = json.loads(sys.argv[1])
        numWolves = 1
        numSheeps = 1
        numMasters = 1
        numDistractor = 3
        numAgent = numWolves + numSheeps + numMasters + numDistractor
        numAgentFirstEnv = numWolves + numSheeps
        numAgentSecondEnv = numWolves + numSheeps + numMasters
        # numDistractor = 2
        damping = 0.5
        frictionloss = float(condition['frictionloss'])
        masterForce = float(condition['masterForce'])
        killZoneRatio = float(condition['killZone'])
        distractKillZoneRatio = float(condition['killZoneofDistractor'])
        ropePunishWeight = float(condition['ropePunishWeight'])
        ropeLength = float(condition['ropeLength'])
        masterMass = float(condition['masterMass'])
        # masterPunishRange = float(condition['masterPunishRange'])
        # masterPullDistance = float(condition['forceAllowedDistance'])
        # masterPullPunish = float(condition['forceForbiddenPunish'])
        masterPullForce = float(condition['masterPullForce'])
        sheepPunishRange = float(condition['sheepPunishRange'])
        sheepForce = float(condition['sheepForce'])
        # masterPullDistanceForSheep = float(condition['forceAllowedDistanceForSheep'])
        wolfForce = float(condition['wolfForce'])
        # wolfFreeDistance = float(condition['wolfFreeDistance'])
        # wolfFreeForce = float(condition['wolfFreeForce'])
        distractorForce = float(condition['distractorForce'])
        maxTimeStep = 25
        visualize=False
        saveAllmodels = True
    print("maddpg: {} wolves, {} sheep, {} blocks, {} episodes with {} steps each eps,  save all models: {}".
          format(numWolves, numSheeps, numMasters, maxEpisode, maxTimeStep,  str(saveAllmodels)))
    print(damping,frictionloss,masterForce)

    dataFolder = os.path.join(dirName, '..','..', 'data')
    mainModelFolder = os.path.join(dataFolder,'modelplus')
    modelFolder = os.path.join(mainModelFolder, 'trainDistractorKillZone','distractorForce={},distractKillZoneRatio={}'.format(distractorForce,distractKillZoneRatio))
    modelFolderForWolfAndSheep = os.path.join(mainModelFolder, 'trainWolfAndSheep',
                                              'frictionloss={},sheepForce={},wolfForce={}'.format(frictionloss,sheepForce, wolfForce))
    fileNameForWolfAndSheep = "maddpg{}episodes{}step_agent".format(maxEpisode, maxTimeStep)
    modelPathsForWolfAndSheep = [os.path.join(modelFolderForWolfAndSheep, fileNameForWolfAndSheep + str(i) + str(maxEpisode) + 'eps') for i in range(numAgentFirstEnv)]

    modelFolderMaster = os.path.join(mainModelFolder, 'trainMaster',
                                              'masterForce={}'.format(masterForce))
    fileNameMaster = "maddpg{}episodes{}step_agent".format(maxEpisode, maxTimeStep)
    modelPathsMaster = [os.path.join(modelFolderMaster, fileNameMaster + str(numAgentSecondEnv-1) + str(maxEpisode) + 'eps')]

    if not os.path.exists(modelFolder):
        os.makedirs(modelFolder)

    wolvesID = [0]
    sheepsID = [1]
    masterID = [2]
    distractorID = [3, 4, 5]
    numKnots = 9
    ropePartIndex = list(range(numAgent, numAgent+numKnots))

    wolfSize = 0.05 #0.075
    sheepSize =  0.05 #0.075
    masterSize =  0.05 #0.075
    distractorSize = 0.05 #0.075
    force = 1.0
    masterRealForce = 15.0
    masterRealPullDistance = 0.4
    forceSigma = 1.0
    knotSize=0

    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [masterSize] * numMasters + [distractorSize] * numDistractor + [knotSize] * numKnots


    killZone = wolfSize * killZoneRatio
    distractKillZone = wolfSize * distractKillZoneRatio
    isCollision = IsCollision(getPosFromAgentState, killZone)
    isCollisionForDistractor = IsCollision(getPosFromAgentState, distractKillZone)
    sheepPunishForOutOfBound = PunishForOutOfBoundVarRange(sheepPunishRange)
    punishForOutOfBound = PunishForOutOfBound()
    zeroPunishForOutOfBound = lambda agentPos:0
    reshapeActionList = [ReshapeAction(wolfForce),ReshapeAction(sheepForce),ReshapeAction(masterRealForce),ReshapeAction(distractorForce),ReshapeAction(distractorForce),ReshapeAction(distractorForce)]
    # rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision, sheepPunishForOutOfBound)
#     punishRope = RewardSheep(ropePartIndex, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision, zeroPunishForOutOfBound)
# #
#     rewardSheepWithRopePunish = lambda state, action, nextState:[sheepRewrad  + ropePunish * ropePunishWeight for sheepRewrad,ropePunish in zip(rewardSheep( state, action, nextState),punishRope( state, action, nextState))]
#     boundForWolfself = BoundForWolfself(wolfFreeDistance, wolfFreeForce)
#     rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList,getPosFromAgentState, isCollision, boundForWolfself)
    punishRopeWithDistractor1 = RewardSheep(ropePartIndex, [distractorID[0]], entitiesSizeList, getPosFromAgentState, isCollision, zeroPunishForOutOfBound,reshapeActionList)
    punishRopeWithDistractor2 = RewardSheep(ropePartIndex, [distractorID[1]], entitiesSizeList, getPosFromAgentState, isCollision, zeroPunishForOutOfBound, reshapeActionList)
    punishRopeWithDistractor3 = RewardSheep(ropePartIndex, [distractorID[2]], entitiesSizeList, getPosFromAgentState, isCollision, zeroPunishForOutOfBound, reshapeActionList)
    rewardDistractor1 = RewardSheep(wolvesID+sheepsID+masterID+[distractorID[1]]+[distractorID[2]], [distractorID[0]], entitiesSizeList, getPosFromAgentState, isCollisionForDistractor,sheepPunishForOutOfBound, reshapeActionList)
    rewardDistractor2 = RewardSheep(wolvesID+sheepsID+masterID+[distractorID[0]]+[distractorID[2]], [distractorID[1]], entitiesSizeList, getPosFromAgentState, isCollisionForDistractor,sheepPunishForOutOfBound, reshapeActionList)
    rewardDistractor3 = RewardSheep(wolvesID+sheepsID+masterID+[distractorID[0]]+[distractorID[1]], [distractorID[2]], entitiesSizeList, getPosFromAgentState, isCollisionForDistractor,sheepPunishForOutOfBound, reshapeActionList)
    rewardDistractorWithRopePunish1 = lambda state, action, nextState:[sheepRewrad  + ropePunish * ropePunishWeight for sheepRewrad,ropePunish in zip(rewardDistractor1( state, action, nextState),punishRopeWithDistractor1( state, action, nextState))]
    rewardDistractorWithRopePunish2 = lambda state, action, nextState:[sheepRewrad  + ropePunish * ropePunishWeight for sheepRewrad,ropePunish in zip(rewardDistractor2( state, action, nextState),punishRopeWithDistractor2( state, action, nextState))]
    rewardDistractorWithRopePunish3 = lambda state, action, nextState:[sheepRewrad  + ropePunish * ropePunishWeight for sheepRewrad,ropePunish in zip(rewardDistractor3( state, action, nextState),punishRopeWithDistractor3( state, action, nextState))]
    # masterPunishForOutOfBound=PunishForOutOfBoundVarRange(masterPunishRange)
    # masterBoundForWolves = BoundForWolves(masterPullDistance, masterPullDistanceForSheep, masterPullPunish, masterPullForce)
    # rewardMaster= RewardMaster(masterID, wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision, masterPunishForOutOfBound, masterBoundForWolves, collisionPunishment=0)

    rewardFunc = lambda state, action, nextState: \
        [1] + [1] + [1] + list(rewardDistractorWithRopePunish1(state, action, nextState)) + list(rewardDistractorWithRopePunish2(state, action, nextState)) + list(rewardDistractorWithRopePunish3(state, action, nextState))

    physicsDynamicsPath=os.path.join(dirName,'..','..','env','xml','leased2Distractor_masterMass={}_ropeLength={}.xml'.format(masterMass,ropeLength))
    print('loadEnv:{}'.format(physicsDynamicsPath))
    with open(physicsDynamicsPath) as f:
        xml_string = f.read()

    makePropertyList=MakePropertyList(transferNumberListToStr)

    geomIds=[1,2,3,4,5,6]
    keyNameList=[0,1]
    valueList=[[damping,damping]]*len(geomIds)
    dampngParameter=makePropertyList(geomIds,keyNameList,valueList)
    changeJointDampingProperty=lambda envDict,geomPropertyDict:changeJointProperty(envDict,geomPropertyDict,'@damping')

    geomIds=[1,2,3,4,5,6]
    keyNameList=[0,1]
    valueList=[[frictionloss,frictionloss]]*len(geomIds)
    frictionlossParameter=makePropertyList(geomIds,keyNameList,valueList)
    changeJointFrictionlossProperty=lambda envDict,geomPropertyDict:changeJointProperty(envDict,geomPropertyDict,'@frictionloss')

    envXmlDict = xmltodict.parse(xml_string.strip())
    envXmlPropertyDictList=[dampngParameter,frictionlossParameter]
    changeEnvXmlPropertFuntionyList=[changeJointDampingProperty,changeJointFrictionlossProperty]
    for propertyDict,changeXmlProperty in zip(envXmlPropertyDictList,changeEnvXmlPropertFuntionyList):
        envXmlDict=changeXmlProperty(envXmlDict,propertyDict)

    envXml=xmltodict.unparse(envXmlDict)
    physicsModel = mujoco.load_model_from_xml(envXml)
    physicsSimulation = mujoco.MjSim(physicsModel)

    numKnots = 9
    numAxis = (numKnots + numAgent) * 2
    qPosInit = (0, ) * numAxis
    qVelInit = (0, ) * numAxis
    qPosInitNoise = 0.6
    qVelInitNoise = 0
    tiedAgentId = [0, 2]
    ropePartIndex = list(range(numAgent, numAgent+numKnots))
    maxRopePartLength = ropeLength
    reset = ResetUniformWithoutXPosForLeashed(physicsSimulation, qPosInit, qVelInit, numAgent, tiedAgentId,ropePartIndex, maxRopePartLength, qPosInitNoise, qVelInitNoise)

    numSimulationFrames=10
    isTerminal= lambda state: False

    transit=TransitionFunctionWithoutXPos(physicsSimulation, numSimulationFrames, visualize,isTerminal, reshapeActionList)

    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID + masterID + distractorID, [], getPosFromAgentState,getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgent)]
    initObsForParams = observe(reset())
    print(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]

    observeOneAgentMaster = lambda agentID: Observe(agentID, wolvesID, sheepsID + masterID, [], getPosFromAgentState,getVelFromAgentState)
    observeMaster = lambda state: [observeOneAgentMaster(agentID)(state) for agentID in range(numAgentSecondEnv)]
    initObsForParamsMaster = observeMaster(reset())
    # print(reset())
    obsShapeMaster = [initObsForParamsMaster[obsID].shape[0] for obsID in range(len(initObsForParamsMaster))]

    observeOneAgentWolfAndSheep = lambda agentID: Observe(agentID, wolvesID, sheepsID, [], getPosFromAgentState, getVelFromAgentState)
    observeWolfAndSheep = lambda state: [observeOneAgentWolfAndSheep(agentID)(state) for agentID in range(numAgentFirstEnv)]
    initObsForParamsWolfAndSheep = observeWolfAndSheep(reset())
    obsShapeWolfAndSheep = [initObsForParamsWolfAndSheep[obsID].shape[0] for obsID in range(len(initObsForParamsWolfAndSheep))]

    print('24e',obsShape)
    print('24eWolfAndSheep',obsShapeWolfAndSheep)
    worldDim = 2
    actionDim = worldDim * 2 + 1

    layerWidth = [128, 128]

#------------ models ------------------------
    buildMADDPGModelsForWolfAndSheep = BuildMADDPGModels(actionDim, numAgentFirstEnv, obsShapeWolfAndSheep)
    modelsListForWolfAndSheep = [buildMADDPGModelsForWolfAndSheep(layerWidth, agentID) for agentID in range(numAgentFirstEnv)]
    [restoreVariables(model, path) for model, path in zip(modelsListForWolfAndSheep, modelPathsForWolfAndSheep)]

    buildMADDPGModelsForMaster = BuildMADDPGModels(actionDim, numAgentSecondEnv, obsShapeMaster)
    modelsListForMaster = [buildMADDPGModelsForMaster(layerWidth, agentID) for agentID in range(numAgentSecondEnv)]
    [restoreVariables(model, path) for model, path in zip([modelsListForMaster[2]], modelPathsMaster)]

    buildMADDPGModelsForDistractor = BuildMADDPGModels(actionDim, numAgent, obsShape)
    modelsListForDistractor = [buildMADDPGModelsForDistractor(layerWidth, agentID) for agentID in range(numAgent)]

    trainCriticBySASR = TrainCriticBySASR(actByPolicyTargetNoisyForNextState, learningRateCritic, gamma)
    trainCritic = TrainCritic(trainCriticBySASR)
    trainActorFromSA = TrainActorFromSA(learningRateActor)
    trainActor = TrainActor(trainActorFromSA)

    paramUpdateInterval = 1 #
    updateParameters = UpdateParameters(paramUpdateInterval, tau)
    sampleBatchFromMemory = SampleFromMemory(minibatchSize)

    learnInterval = 100
    learningStartBufferSize = minibatchSize * maxTimeStep
    startLearn = StartLearn(learningStartBufferSize, learnInterval)

    trainMADDPGModels = TrainMADDPGModelsWithBuffer(updateParameters, trainActor, trainCritic, sampleBatchFromMemory, startLearn, modelsListForDistractor)

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    actOneStepOneModelRecreate = ActOneStepRecreate(actByPolicyTrainNoisy, masterID,wolvesID, sheepsID, getPosFromAgentState, masterRealPullDistance, force, forceSigma)
    actOneStepForWolfAndSheep = lambda allAgentsStates, runTime: [actOneStepOneModel(model, allAgentsStates) for model in modelsListForWolfAndSheep]
    actOneStepForMaster = lambda allAgentsStates, state: [actOneStepOneModelRecreate(modelsListForMaster[2], allAgentsStates, state)]
    actOneStepForDistractor = lambda allAgentsStates, runTime: [actOneStepOneModel(model, allAgentsStates) for model in modelsListForDistractor[3:6]]
    sampleOneStep = SampleOneStep(transit, rewardFunc)
    runDDPGTimeStep = RunTimeStep(actOneStepForWolfAndSheep, actOneStepForMaster, actOneStepForDistractor, sampleOneStep, trainMADDPGModels, observeWolfAndSheep = observeWolfAndSheep, observeMaster = observeMaster, observe = observe)

    runEpisode = RunEpisode(reset, runDDPGTimeStep, maxTimeStep, isTerminal)

    getAgentModel = lambda agentId: lambda: trainMADDPGModels.getTrainedModels()[agentId]
    getModelList = [getAgentModel(i) for i in distractorID]

    modelSaveRate = 1000
    fileName = "maddpg{}episodes{}step_agent".format(maxEpisode, maxTimeStep)

    modelPath = os.path.join(modelFolder, fileName)

    saveModels = [SaveModel(modelSaveRate, saveVariables, getTrainedModel, modelPath + str(i+numAgentSecondEnv), saveAllmodels) for i, getTrainedModel in enumerate(getModelList)]

    maddpg = RunAlgorithm(runEpisode, maxEpisode, saveModels, numAgent)
    replayBuffer = getBuffer(bufferSize)
    meanRewardList, trajectory = maddpg(replayBuffer)





if __name__ == '__main__':
    main()


