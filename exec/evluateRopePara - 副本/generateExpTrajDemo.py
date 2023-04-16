import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import xmltodict
import mujoco_py as mujoco

import itertools as it
from collections import OrderedDict
import numpy as np
from env.multiAgentMujocoEnv import TransitionFunctionWithoutXPosForExp, RewardSheep, RewardWolf, Observe, IsCollision, getPosFromAgentState, \
    getVelFromAgentState,PunishForOutOfBound,ReshapeAction, TransitionFunctionWithoutXPos, ResetUniformWithoutXPosForLeashed, PunishForOutOfBoundVarRange,BoundForWolfself

from src.maddpg.trainer.myMADDPG import ActOneStep, BuildMADDPGModels, actByPolicyTrainNoisy, ActOneStepRecreate
from src.functionTools.loadSaveModel import saveToPickle, restoreVariables,GetSavePath, loadFromPickle, ScaleTrajectory
from src.functionTools.trajectory import SampleExpTrajectory,SampleExpTrajectoryWithAllFrames,SampleExpTrajectoryWithAllFramesRecreate
from src.functionTools.editEnvXml import transferNumberListToStr,MakePropertyList,changeJointProperty
from src.visualize.visualizeMultiAgent import Render



wolfColor = np.array([0.85, 0.35, 0.35])
sheepColor = np.array([0.35, 0.85, 0.35])
masterColor= np.array([0.35, 0.35, 0.85])
distractorColor = np.array([0.35, 0.85, 0.85])
blockColor = np.array([0.25, 0.25, 0.25])




def generateSingleCondition(condition):
    debug = 0
    if debug:


        damping=2.0
        frictionloss=0.0
        masterForce=1.0

        numWolves = 1
        numSheeps = 1
        numMasters = 1
        numDistractor = 1
        maxTimeStep = 25

        maxEpisode = 60000
        saveTraj=True
        saveImage=True
        visualizeMujoco=False
        visualizeTraj = True
        makeVideo=True
    else:

        # print(sys.argv)
        # condition = json.loads(sys.argv[1])
        damping = 0.5
        frictionloss = 0.0
        offset = 0.0
        killZoneRatio = 2.0
        # distractKillZoneRatio = 0.0
        ropePunishWeight = 0.3
        ropeLength = 0.05
        masterMass = 1.0
        masterForce = float(condition['masterForce'])
        masterPunishRange = 0.5
        masterPullDistance = float(condition['forceAllowedDistance'])
        masterPullPunish = 1.0
        masterPullForce = float(condition['masterPullForce'])
        sheepPunishRange = float(condition['sheepPunishRange'])
        sheepForce = 5.0
        masterPullDistanceForSheep = float(condition['forceAllowedDistanceForSheep'])
        masterForcePunishAmplify = float(condition['masterForcePunishAmplify'])
        wolfForce = 4.0
        distractorNoise = float(condition['distractorNoise'])
        wolfFreeDistance = float(condition['wolfFreeDistance'])
        distractorForce = float(condition['distractorForce'])
        wolfFreeForce = 7.0
        masterPunishAmplify = 1.0
        viscosity = 0.00002
        dt = 0.02
        offsetFrame = int (offset/dt)

        force = 1.0
        masterRealForce = 20.0
        masterRealPullDistance = 0.4
        noiseDistractor=True
        if noiseDistractor:
            distractorNoise = 16.0

        maxEpisode = 90000
        evaluateEpisode = 90000
        numWolves = 1
        numSheeps = 1
        numMasters = 1
        numDistractor = 2


        # numDistractor = 2
        maxTimeStep = 25
        numAgentFirstEnv = numWolves + numSheeps
        numAgentSecondEnv = numWolves + numSheeps + numMasters
        numAgentThirdEnv = numWolves + numSheeps + numMasters + numDistractor
        # numAgentThirdEnv = numWolves + numSheeps + numMasters + numDistractor
        # noiseDistractor=False
        # if noiseDistractor:
        #     distractorNoise = 32.0

        saveTraj=True
        saveImage=True
        visualizeMujoco=False
        visualizeTraj = True
        makeVideo=True

    evalNum = 1
    maxRunningStepsToSample = 1000
    modelSaveName = 'trainDistractorFinalNoiseDemo'
    dataFolder = os.path.join(dirName, '..','..', 'data')
    mainModelFolder = os.path.join(dataFolder,'modelplus')

    modelFolderForWolfAndSheep = os.path.join(mainModelFolder, 'trainWolfAndSheep',
                                              'frictionloss={},sheepForce={},wolfForce={}'.format(frictionloss,sheepForce, wolfForce))
    fileNameForWolfAndSheep = "maddpg{}episodes{}step_agent".format(maxEpisode, maxTimeStep)
    modelPathsForWolfAndSheep = [os.path.join(modelFolderForWolfAndSheep, fileNameForWolfAndSheep + str(i) + str(maxEpisode) + 'eps') for i in range(numAgentFirstEnv)]

    modelFolderMaster = os.path.join(mainModelFolder, 'trainMaster','masterForce={}'.format(masterForce))
    fileNameMaster = "maddpg{}episodes{}step_agent".format(maxEpisode, maxTimeStep)
    modelPathsMaster = [os.path.join(modelFolderMaster, fileNameMaster + str(numAgentSecondEnv-1) + str(maxEpisode) + 'eps')]

    modelFolderForDistractor = os.path.join(mainModelFolder, 'trainDistractorFinalNoDisturb','distractorForce={}'.format(distractorForce))
    fileNameForDistractor = "maddpg{}episodes{}step_agent".format(maxEpisode, maxTimeStep)
    modelPathsForDistractor = [os.path.join(modelFolderForDistractor, fileNameForDistractor + str(i+numAgentSecondEnv) + str(maxEpisode) + 'eps') for i in range(numDistractor)]
    demoTrajFolder = os.path.join(dirName, '..','..', 'data')
    demoTrajPath = os.path.join(demoTrajFolder, 'trajDemo',
                                              'evalNum={}_forceAllowedDistanceForSheep={}_masterForce={}_masterPullForce={}.pickle'.format(20, 0.4, 20.0, 15.0))
    demoTraj = loadFromPickle(demoTrajPath)
    demoTrajFolderRep = os.path.join(dirName, '..','..', 'data')
    demoTrajPathRep = os.path.join(demoTrajFolderRep, 'trajDemo',
                                              'evalNum={}_forceAllowedDistanceForSheep={}_masterForce={}_masterPullForce={}_offset={}.pickle'.format(20, 0.4, 20.0, 15.0,0.0))
    demoTrajRep = loadFromPickle(demoTrajPathRep)

    # modelFolderForDistractor = os.path.join(mainModelFolder, 'trainDistractor','ropeLength={},masterForce={}'.format(ropeLength,masterForce))
    # fileNameForDistractor = "maddpg{}episodes{}step_agent".format(maxEpisode, maxTimeStep)
    # modelPathsForDistractor = [os.path.join(modelFolderForDistractor, fileNameForDistractor + str(i+numAgentSecondEnv) + str(maxEpisode) + 'eps') for i in range(numDistractor)]

    # print("maddpg: , saveTraj: {}, visualize: {},damping; {},frictionloss: {}".format( str(saveTraj), str(visualizeMujoco),damping,frictionloss))
    wolvesID = [0]
    sheepsID = [1]
    masterID = [2]
    distractorID = [3,4]
    hideIdList = [1,3]# distractorID + sheepsID


    numAgent = numWolves + numSheeps + numMasters +  numDistractor
    numAgentForDarw =  3



    wolfSize = 0.05
    sheepSize = 0.05
    masterSize = 0.05
    distractorSize = 0.05
    positionIndex = [0, 1]
    FPS = 40
    rawXRange = [-1, 1]
    rawYRange = [-1, 1]
    # rawXRange = [200, 600]
    # rawYRange = [200, 600]
    scaledXRange = [-0.8, 0.8]
    scaledYRange = [-0.8, 0.8]
    scaleTrajectoryInSpace = ScaleTrajectory(positionIndex, rawXRange, rawYRange, scaledXRange, scaledYRange)
    entitiesSizeList = [wolfSize] * numWolves + [masterSize] * numMasters + [distractorSize] * 1
    entitiesSizeListForDarw = [wolfSize] * numWolves + [masterSize] * numMasters + [distractorSize] * 1

    entitiesMovableList = [True] * numAgent + [False] * numMasters
    noiseMean = (0, 0)
    noiseCov = [[distractorNoise, 0], [0, distractorNoise]]
    distractorReshapeAction=ReshapeAction(5)
    noiseDistractorAction= lambda state:limitForceMagnitude((distractorReshapeAction(state)+np.random.multivariate_normal(noiseMean, noiseCov, (1, 1), 'raise')[0])[0])
    if noiseDistractor:
          reshapeActionList = [ReshapeAction(wolfForce),ReshapeAction(sheepForce),ReshapeAction(masterRealForce),noiseDistractorAction,noiseDistractorAction]
    else:
        reshapeActionList = [ReshapeAction(wolfForce),ReshapeAction(sheepForce),ReshapeAction(masterRealForce),ReshapeAction(distractorForce),ReshapeAction(distractorForce)]
    # reshapeActionList = [ReshapeAction(wolfForce),ReshapeAction(sheepForce),ReshapeAction(masterRealForce),ReshapeAction(distractorForce),ReshapeAction(distractorForce)]
    killZone = wolfSize * killZoneRatio
    isCollision = IsCollision(getPosFromAgentState, killZone)
    sheepPunishForOutOfBound = PunishForOutOfBoundVarRange(sheepPunishRange)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision, sheepPunishForOutOfBound, reshapeActionList)
    boundForWolfself = BoundForWolfself(wolfFreeDistance, wolfFreeForce)
    rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList,getPosFromAgentState, isCollision, boundForWolfself)
    # rewardDistractor = RewardSheep(wolvesID+sheepsID+masterID, distractorID, entitiesSizeList, getPosFromAgentState, isCollision,punishForOutOfBound)
    rewardMaster= lambda state, action, nextState: [-reward  for reward in rewardWolf(state, action, nextState)]
    rewardFunc = lambda state, action, nextState: \
        list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))\
        + list(rewardMaster(state, action, nextState))

    physicsDynamicsPath=os.path.join(dirName,'..','..','env','xml','leased2Distractor_masterMass={}_ropeLength={}.xml'.format(masterMass,ropeLength))
    print('loadEnv:{}'.format(physicsDynamicsPath))
    with open(physicsDynamicsPath) as f:
        xml_string = f.read()

    makePropertyList=MakePropertyList(transferNumberListToStr)

    geomIds=[1,2,3,4,5]
    keyNameList=[0,1]
    valueList=[[damping,damping]]*len(geomIds)
    dampngParameter=makePropertyList(geomIds,keyNameList,valueList)

    changeJointDampingProperty=lambda envDict,geomPropertyDict:changeJointProperty(envDict,geomPropertyDict,'@damping')

    geomIds=[1,2,3,4,5]
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
    qPosInitNoise = 0.4
    qVelInitNoise = 0
    tiedAgentId = [0, 2]
    ropePartIndex = list(range(numAgent, numAgent + numKnots))
    maxRopePartLength = ropeLength
    reset = ResetUniformWithoutXPosForLeashed(physicsSimulation, qPosInit, qVelInit, numAgent, tiedAgentId,ropePartIndex, maxRopePartLength, qPosInitNoise, qVelInitNoise)
    numSimulationFrames=1
    isTerminal= lambda state: False
    # distractorReshapeAction=ReshapeAction(5)
    noiseMean = (0, 0)
    # noiseCov = [[distractorNoise, 0], [0, distractorNoise]]
    # x = np.random.multivariate_normal(noiseMean, noiseCov, (1, 1), 'raise')[0]
    class LimitForceMagnitude():
        def __init__(self,entityMaxForceMagnitude=None):
            self.entityMaxForceMagnitude = entityMaxForceMagnitude

        def __call__(self,entityNextForce):
            # print(entityNextForce)
            if self.entityMaxForceMagnitude is not None:
                forceMagnitude = np.sqrt(np.square(entityNextForce[0]) + np.square(entityNextForce[1])) #
            if forceMagnitude > self.entityMaxForceMagnitude:
                entityNextForce = entityNextForce / forceMagnitude * self.entityMaxForceMagnitude

            return np.array(entityNextForce)

    limitForceMagnitude = LimitForceMagnitude(5)

    # noiseDistractorAction= lambda state:limitForceMagnitude((distractorReshapeAction(state)+np.random.multivariate_normal(noiseMean, noiseCov, (1, 1), 'raise')[0])[0])
    # if noiseDistractor:
    #       reshapeActionList = [ReshapeAction(wolfForce),ReshapeAction(sheepForce),ReshapeAction(masterForce),noiseDistractorAction,noiseDistractorAction]
    # else:
    #     reshapeActionList = [ReshapeAction(wolfForce),ReshapeAction(sheepForce),ReshapeAction(masterForce),ReshapeAction(15),ReshapeAction(15)]




    transit=TransitionFunctionWithoutXPosForExp(physicsSimulation, numSimulationFrames, visualizeMujoco,isTerminal, reshapeActionList)


    sampleTrajectory = SampleExpTrajectoryWithAllFramesRecreate(maxRunningStepsToSample, transit, isTerminal, reset)


    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID + masterID +distractorID, [], getPosFromAgentState, getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgent)]
    # print(reset())

    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]
    # print('24e',obsShape)

    observeOneAgentMaster = lambda agentID: Observe(agentID, wolvesID, sheepsID + masterID, [], getPosFromAgentState,getVelFromAgentState)
    observeMaster = lambda state: [observeOneAgentMaster(agentID)(state) for agentID in range(numAgentSecondEnv)]
    initObsForParamsMaster = observeMaster(reset())
    obsShapeMaster = [initObsForParamsMaster[obsID].shape[0] for obsID in range(len(initObsForParamsMaster))]

    observeOneAgentWolfAndSheep = lambda agentID: Observe(agentID, wolvesID, sheepsID, [], getPosFromAgentState, getVelFromAgentState)
    observeWolfAndSheep = lambda state: [observeOneAgentWolfAndSheep(agentID)(state) for agentID in range(numAgentFirstEnv)]
    initObsForParamsWolfAndSheep = observeWolfAndSheep(reset())
    obsShapeWolfAndSheep = [initObsForParamsWolfAndSheep[obsID].shape[0] for obsID in range(len(initObsForParamsWolfAndSheep))]

    worldDim = 2
    actionDim = worldDim * 2 + 1

    layerWidth = [128, 128]

    # ------------ model ------------------------
    buildMADDPGModelsForWolfAndSheep = BuildMADDPGModels(actionDim, numAgentFirstEnv, obsShapeWolfAndSheep)
    modelsListForWolfAndSheep = [buildMADDPGModelsForWolfAndSheep(layerWidth, agentID) for agentID in range(numAgentFirstEnv)]
    [restoreVariables(model, path) for model, path in zip(modelsListForWolfAndSheep, modelPathsForWolfAndSheep)]

    buildMADDPGModelsForMaster = BuildMADDPGModels(actionDim, numAgentSecondEnv, obsShapeMaster)
    modelsListForMaster = [buildMADDPGModelsForMaster(layerWidth, agentID) for agentID in range(numAgentSecondEnv)]
    [restoreVariables(model, path) for model, path in zip([modelsListForMaster[2]], modelPathsMaster)]

    buildMADDPGModelsForDistractor = BuildMADDPGModels(actionDim, numAgentThirdEnv, obsShape)
    modelsListForDistractor = [buildMADDPGModelsForDistractor(layerWidth, agentID) for agentID in range(numAgentThirdEnv)]
    [restoreVariables(model, path) for model, path in zip(modelsListForDistractor[3:5], modelPathsForDistractor)]
    modelList = modelsListForWolfAndSheep + [modelsListForMaster[2]]

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    actOneStepOneModelRecreate = ActOneStepRecreate(actByPolicyTrainNoisy, masterID,wolvesID, sheepsID, getPosFromAgentState, masterRealPullDistance, force)
    actOneStepForWolfAndSheep = lambda allAgentsStates: [actOneStepOneModel(model, allAgentsStates) for model in modelsListForWolfAndSheep]
    actOneStepForMaster = lambda allAgentsStates, state: [actOneStepOneModelRecreate(modelsListForMaster[2], allAgentsStates, state)]
    actOneStepForDistractor = lambda allAgentsStates: [actOneStepOneModel(model, allAgentsStates) for model in modelsListForDistractor[3:5]]
    # offsetFrameList=[0] + [offsetFrame]*3
    offsetFrameList=[offsetFrame,offsetFrame,0] #wolf sheep master
    for hideId in hideIdList:
        agentList = list(range(numAgent))
        del(agentList[hideId])
        trajList = []
        expTrajList = []
        newTrajList = []
        for _ in range(evalNum):
            # np.random.seed(i)
            traj, expTraj = sampleTrajectory(actOneStepForWolfAndSheep, actOneStepForMaster, actOneStepForDistractor, observeWolfAndSheep, observeMaster, observe, demoTraj)
            trajList.append(list(traj))
            expTrajList.append((list(expTraj)))
        for i,traj in enumerate(expTrajList):
            newTraj = [[state[agentId] for agentId in agentList]  for state in traj]
            if offsetFrame < 0:
                offsetTraj =  [[newTraj[index+offsetF][i] for i,offsetF in enumerate(offsetFrameList)]  for index in range(-offsetFrame,len(newTraj))]
            else:
                offsetTraj =  [[newTraj[index+offsetF][i] for i,offsetF in enumerate(offsetFrameList)]  for index in range(len(newTraj)-offsetFrame)]
            newTrajList.append(offsetTraj)

        print('save',newTrajList[0][0])
        # saveTraj
        if saveTraj:
            # trajFileName = "maddpg{}wolves{}sheep{}blocks{}eps{}step{}Traj".format(numWolves, numSheeps, numMasters, maxEpisode, maxTimeStep)

            trajectoriesSaveDirectory= os.path.join(dataFolder,'trajectorySeperatedNoDistactor',modelSaveName,'ENDnoiseOffsetWithMasterPull')
            if not os.path.exists(trajectoriesSaveDirectory):
                os.makedirs(trajectoriesSaveDirectory)

            trajectorySaveExtension = '_forexp.pickle'
            fixedParameters = {'masterForce':masterRealForce,'masterPullForce':masterPullForce,'forceAllowedDistanceForSheep':masterRealPullDistance}
            generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
            trajectorySavePath = generateTrajectorySavePath(condition)
            saveToPickle(trajList, trajectorySavePath)

            expTrajectoriesSaveDirectory = os.path.join(dataFolder, 'ExptrajectorySeperatedNoDistactor', modelSaveName,'ENDnoiseOffsetWithMasterPull')
            if not os.path.exists(expTrajectoriesSaveDirectory):
                os.makedirs(expTrajectoriesSaveDirectory)

            generateExpTrajectorySavePath = GetSavePath(expTrajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
            expTrajectorySavePath = generateExpTrajectorySavePath(condition)
            saveToPickle(newTrajList, expTrajectorySavePath)
        for i in range(1000):
            newTrajList[0][i][0] = demoTrajRep[0][i][0]
            newTrajList[0][i][1] = demoTrajRep[0][i][2]
        # visualize
        if visualizeTraj:

            # pictureFolder = os.path.join(dataFolder, 'demo', modelSaveName,'normal','damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))
            pictureFolder = os.path.join(dataFolder, 'demo', modelSaveName,'REP2distractorForce={}'.format(distractorForce))
            if not os.path.exists(pictureFolder):
                os.makedirs(pictureFolder)
            else:
                return 1
            entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [masterColor] * numMasters + [distractorColor] * numDistractor
            render = Render(entitiesSizeListForDarw, entitiesColorList, numAgentForDarw,pictureFolder,saveImage, getPosFromAgentState)
            trajToRenderResize = scaleTrajectoryInSpace(newTrajList)
            trajToRender = np.concatenate(trajToRenderResize)
            print(np.size(trajToRender,0))
            render(trajToRender)


def main():

    manipulatedVariables = OrderedDict()
    # manipulatedVariables['damping'] = [0.5]#[0.0, 1.0]
    # manipulatedVariables['frictionloss'] =[1.0]# [0.0, 0.2, 0.4]
    # manipulatedVariables['masterForce'] = [10.0]
    # manipulatedVariables['offset'] = [-2,-1,-0.5, 0 ,0.5,1,2]
    # manipulatedVariables['distractorNoise']=[3.0]


 #ssr-1,Xp = 0.06; ssr-3 =0.09
 #ssr-1, ssr-3 = 1.0; Xp = 2.0

    manipulatedVariables['forceAllowedDistance'] = [0.3]
    manipulatedVariables['sheepPunishRange'] = [0.6]
    manipulatedVariables['masterForce'] = [30.0]
    manipulatedVariables['forceAllowedDistanceForSheep'] = [0.3]
    manipulatedVariables['masterForcePunishAmplify'] = [10]
    manipulatedVariables['distractorNoise'] = [20.0]
    manipulatedVariables['wolfFreeDistance'] = [1.2]
    manipulatedVariables['masterPullForce'] = [15.0]
    manipulatedVariables['distractorForce'] = [5.0]
    # manipulatedVariables['viscosity'] = [0.00002, 0.0002]
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    conditions = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]
    for condition in conditions:
        # print(condition)
        generateSingleCondition(condition)
        # try:
            # generateSingleCondition(condition)
        # except:
            # continue

if __name__ == '__main__':
    main()
                                                 
