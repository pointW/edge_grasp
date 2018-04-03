import os
import openravepy
import point_cloud
from numpy import dot, array, eye, vstack
from numpy.linalg import inv


class Simulator:
    def __init__(self):
        projectDir = os.getcwd() + "/../"
        self.env = openravepy.Environment()
        self.env.SetViewer('qtcoin')
        self.env.Load(projectDir + "openrave/environment_2.xml")
        self.robot = self.env.GetRobots()[0]
        self.robot.SetDOFValues(array([0.0475]))
        self.sensor = self.env.GetSensors()[0]
        self.sTh = dot(inv(self.sensor.GetTransform()), self.robot.GetTransform())

    def GetCloud(self, workspace=None):
        '''Agent gets point cloud from its sensor from the current position.'''

        self.StartSensor()
        self.env.StepSimulation(0.001)

        data = self.sensor.GetSensorData(openravepy.Sensor.Type.Laser)
        cloud = data.ranges + data.positions[0]

        self.StopSensor()

        if workspace is not None:
            cloud = point_cloud.FilterWorkspace(workspace, cloud)

        return cloud

    def MoveSensorToPose(self, T):
        '''Moves the hand of the robot to the specified pose.'''

        with self.env:
            self.robot.SetTransform(dot(T, self.sTh))
            self.env.UpdatePublishedBodies()

    def StartSensor(self):
        '''Starts the sensor in openrave, displaying yellow haze.'''

        self.sensor.Configure(openravepy.Sensor.ConfigureCommand.PowerOn)
        self.sensor.Configure(openravepy.Sensor.ConfigureCommand.RenderDataOn)

    def StopSensor(self):
        '''Disables the sensor in openrave, removing the yellow haze.'''

        self.sensor.Configure(openravepy.Sensor.ConfigureCommand.PowerOff)
        self.sensor.Configure(openravepy.Sensor.ConfigureCommand.RenderDataOff)


if __name__ == '__main__':
    simulator = Simulator()
    workspace = [(-1, 3), (-2, 2), (-2, 2)]
    T = eye(4)
    T[0:3, 0] = [0, 0, 1]
    T[0:3, 1] = [0, 1, 0]
    T[0:3, 2] = [-1, 0, 0]
    cloud = None
    for T3 in [[1, -1, 0.2],
               [1, -1.5, 0.2],
               [1, -2, 0.2]]:
        T[0:3, 3] = T3

        simulator.MoveSensorToPose(T)

        cloud1 = simulator.GetCloud(workspace)
        if cloud is None:
            cloud = cloud1
        else:
            cloud = vstack([cloud, cloud1])

    file = os.getcwd() + "/../data/testpcd.pcd"
    point_cloud.SavePcd(file, cloud)
    print cloud