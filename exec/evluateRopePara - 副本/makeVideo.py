import os
dirName = os.path.dirname(__file__)

import cv2
from collections import OrderedDict
import itertools as it


def makeVideo(condition):
    debug = 0
    if debug:

        damping=2.0
        frictionloss=0.0
        masterForce=1.0

        numTrajToSample=2
        maxRunningStepsToSample=100
    else:
        # print(sys.argv)
        # condition = json.loads(sys.argv[1])
        # damping = float(condition['damping'])
        # frictionloss = float(condition['frictionloss'])
        # masterForce = float(condition['masterForce'])
        # # distractorNoise = float(condition['distractorNoise'])
        # offsetFrame = int(condition['offsetFrame'])
        # damping = float(condition['damping'])
        # frictionloss = float(condition['frictionloss'])
        masterForce = float(condition['masterForce'])
        offset = 0.0
        frictionloss = float(condition['frictionloss'])
        killZoneRatio = float(condition['killZone'])
        ropePunishWeight = float(condition['ropePunishWeight'])
        ropeLength = float(condition['ropeLength'])
        masterMass = float(condition['masterMass'])
        masterPullDistance = float(condition['forceAllowedDistance'])
        masterPullPunish = float(condition['forceForbiddenPunish'])
        masterPullForce = float(condition['masterPullForce'])
        sheepPunishRange = float(condition['sheepPunishRange'])
        masterPunishRange = float(condition['masterPunishRange'])
        sheepForce = float(condition['sheepForce'])
        masterPullDistanceForSheep = float(condition['forceAllowedDistanceForSheep'])
        wolfForce = float(condition['wolfForce'])
        wolfFreeDistance = float(condition['wolfFreeDistance'])
        wolfFreeForce = float(condition['wolfFreeForce'])
        distractorForce = float(condition['distractorForce'])
        hideId = int(condition['hideId'])
        masterRealPullDistanceWM = float(condition['RealAllowedDistanceWM'])
        sheepKillZoneRatio = float(condition['killZoneofSheep'])
        forceSigma = float(condition['forceSigma'])
        distractKillZoneRatio = float(condition['killZoneofDistractor'])
        dt = 0.02
        offsetFrame = int (offset/dt)
        masterRealPullDistance = 0.4
        distractorNoise = 16.0
        # damping = float(condition['damping'])
        # frictionloss = float(condition['frictionloss'])
        masterForce = float(condition['masterForce'])
        # distractorNoise = float(condition['distractorNoise'])
        # offset = float(condition['offset'])
        # hideId = int(condition['hideId'])

        numTrajToSample=1
        maxRunningStepsToSample=1000


    dataFolder = os.path.join(dirName, '..','..', 'data')
    mainDemoFolder = os.path.join(dataFolder,'demo')
    # videoFolder=os.path.join(mainDemoFolder, 'expTrajMADDPGMujocoEnvWithRopeAdd2Distractors','normal')
    videoFolder=os.path.join(mainDemoFolder, 'trainDistractorKillZoneExpDemo')
    # videoFolder=os.path.join(mainDemoFolder, 'expTrajMADDPGMujocoEnvWithRopeAddDistractor_wolfHideSpeed','CrossSheep',)
    # videoFolder=os.path.join(mainDemoFolder, '2expTrajMADDPGMujocoEnvWithRopeAddDistractor_wolfHideSpeed','noise','NoiseDistractor')
    # videoFolder=os.path.join(mainDemoFolder, '2expTrajMADDPGMujocoEnvWithRopeAddDistractor_wolfHideSpeed','OffsetWolfForward')
    # videoFolder=os.path.join(mainDemoFolder, 'demo', 'expTrajMADDPGMujocoEnvWithRopeAdd2DistractorsWithRopePunish','normal',)
    if not os.path.exists(videoFolder):
        os.makedirs(videoFolder)
    # videoPath= os.path.join(videoFolder,'MADDPGMujocoEnvWithRopeAdd2Distractor_damping={}_frictionloss={}_masterForce={}.avi'.format(damping,frictionloss,masterForce))
    videoPath= os.path.join(videoFolder,'V4ChasingNoConsdistractorForce={},distractKillZoneRatio={}.avi'.format(distractorForce,distractKillZoneRatio))

    # videoPath= os.path.join(mainDemoFolder,'MADDPGMujocoEnvWithRopeAdd2DistractorWithRopePunish_damping={}_frictionloss={}_masterForce={}.avi'.format(damping,frictionloss,masterForce))
    # videoPath= os.path.join(videoFolder,'CrossSheepMADDPGMujocoEnvWithRopeAddDistractor_wolfHideSpeed_damping={}_frictionloss={}_masterForce={}.avi'.format(damping,frictionloss,masterForce))
    # videoPath= os.path.join(videoFolder,'OffsetWolfForwardMADDPGMujocoEnvWithRopeAddDistractor_wolfHideSpeed_damping={}_frictionloss={}_masterForce={}_offsetFrame={}.avi'.format(damping,frictionloss,masterForce,offsetFrame))
    # videoPath = os.path.join(videoFolder,'damping={}_frictionloss={}_masterForce={}_distractorNoise={}.avi'.format(damping,frictionloss,masterForce,distractorNoise))
    # videoPath = os.path.join(videoFolder,'damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))
    print(videoPath)
    fps = 40
    size=(700,700)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    # fourcc = 0
    videoWriter = cv2.VideoWriter(videoPath,fourcc,fps,size)#最后一个是保存图片的尺寸

    #for(i=1;i<471;++i)

    # pictureFolder = os.path.join(videoFolder,'damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))
    # pictureFolder = os.path.join(videoFolder,'damping={}_frictionloss={}_masterForce={}_offsetFrame={}'.format(damping,frictionloss,masterForce,offsetFrame))
    # pictureFolder = os.path.join(videoFolder,'damping={}_frictionloss={}_masterForce={}_distractorNoise={}'.format(damping,frictionloss,masterForce,distractorNoise))
    # pictureFolder = os.path.join(videoFolder,'damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))
    # pictureFolder = os.path.join(videoFolder,'damping={}_frictionloss={}_masterForce={}_offset={}_hideId={}'.format(damping,frictionloss,masterForce,offset,hideId))
    pictureFolder = os.path.join(dataFolder, 'demo', 'trainDistractorKillZoneExpDemo','V4ChasingNoConsdistractorForce={},distractKillZoneRatio={}'.format(distractorForce,distractKillZoneRatio))


    for i in range(0,numTrajToSample*maxRunningStepsToSample):
        imgPath=os.path.join(pictureFolder,'rope'+str(i)+'.png')
        frame = cv2.imread(imgPath)
        img=cv2.resize(frame,size)
        videoWriter.write(img)
    videoWriter.release()

def main():

    manipulatedVariables = OrderedDict()
    # manipulatedVariables['damping'] = [0.5]
    # manipulatedVariables['frictionloss'] = [1.0]
    manipulatedVariables['frictionloss'] = [0.0]
    manipulatedVariables['masterForce'] = [30.0]
    manipulatedVariables['killZone'] = [4.0]
    manipulatedVariables['killZoneofDistractor'] = [0.0]
    manipulatedVariables['ropePunishWeight'] = [0.3]
    manipulatedVariables['ropeLength'] = [0.05] #ssr-1,Xp = 0.06; ssr-3 =0.09
    manipulatedVariables['masterMass'] = [1.0] #ssr-1, ssr-3 = 1.0; Xp = 2.0
    manipulatedVariables['masterPunishRange'] = [0.5] #ssr-1, ssr-3 = 1.0; Xp = 2.0
    # manipulatedVariables['wolfMass'] = [3.0/] #ssr-1, ssr-3 = 1.0; Xp = 2.0
    manipulatedVariables['forceAllowedDistance'] = [0.3]
    manipulatedVariables['forceForbiddenPunish'] = [1.0]
    manipulatedVariables['masterPullForce'] = [18.0]
    manipulatedVariables['sheepPunishRange'] = [0.6]
    manipulatedVariables['sheepForce'] = [5.0]
    manipulatedVariables['forceAllowedDistanceForSheep'] = [1.5]
    manipulatedVariables['wolfForce'] = [4.0]
    manipulatedVariables['wolfFreeDistance'] = [1.2]
    manipulatedVariables['wolfFreeForce'] = [7.0]
    manipulatedVariables['distractorForce'] = [5.0]
    manipulatedVariables['hideId'] = [1]
    manipulatedVariables['RealAllowedDistanceWM'] = [0.4]
    manipulatedVariables['killZoneofSheep'] = [4.0]
    manipulatedVariables['forceSigma'] = [1.0]
    manipulatedVariables['killZoneofDistractor'] = [6.0]
    # manipulatedVariables['offset'] = [0.0]

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    conditions = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    for condition in conditions:
        print(condition)
        makeVideo(condition)
        # try :
            # makeVideo(condition)
        # except :
            # print('error',condition)
#

if __name__ == '__main__':
    main()
