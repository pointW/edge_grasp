'''
This module creates visual markers for rviz.
'''

# system dependencies
import numpy

# ROS dependencies
from geometry_msgs.msg import Point
import rospy
from std_msgs.msg import ColorRGBA
import tf
from tf.transformations import quaternion_from_matrix
from visualization_msgs.msg import Marker, MarkerArray


def createMarker(ns, id, rgb, alpha=1.0, frameName="/base_link", action=Marker.ADD, lifetime=rospy.Duration(60)):
  '''
  Create a visual marker.
  
  @type ns: string
  @param ns: the marker's namspace in rviz
  @type id: integer
  @param id: the marker's ID in rviz
  @type rgb: 1x3 vector
  @param rgb: the marker's color
  @type alpha: the marker's opaquity (1.0 = opaque, 0 = transparent)
  @type frameName: string
  @param frameName: the marker's frame name
  @type action: integer
  @param action: the action that is applied to the marker
  @rtype: rviz visual marker
  @return: the visual marker
  '''
  marker = Marker()
  marker.header.frame_id = frameName
  marker.header.stamp = rospy.get_rostime()
  marker.ns = ns
  marker.id = id  
  marker.action = action
  marker.lifetime = lifetime
  marker.color.r = rgb[0]
  marker.color.g = rgb[1]
  marker.color.b = rgb[2]
  marker.color.a = alpha
  return marker


def createArrowMarker(ns, id, rgb, p, q):
  '''
  Create an arrow marker between two points.
  
  @type ns: string
  @param ns: the marker's namspace in rviz
  @type id: number
  @param id: the marker's ID in rviz
  @type rgb: 1x3 vector
  @param rgb: the marker's color
  @type p: 1x3 vector
  @param p: the point from which the arrow starts
  @type q: 1x3 vector
  @param q: the point at which the arrow ends    
  @rtype: rviz visual marker
  @return: the arrow marker
  '''
  marker = createMarker(ns, id, rgb)
  marker.type = Marker.ARROW
  marker.action = Marker.ADD
  marker.points.append(Point(p[0],p[1],p[2]))
  marker.points.append(Point(q[0],q[1],q[2]))
  marker.scale.x = 0.02
  marker.scale.y = 0.02
  marker.scale.z = 0.02
  return marker
  

def createPointListMarker(ns, id, rgb, pts, indices=[], size=0.03):
  '''
  Create a point-list marker from a list of points.
  
  @type ns: string
  @param ns: the marker's namspace in rviz
  @type id: number
  @param id: the marker's ID in rviz
  @type rgb: 1x3 vector
  @param rgb: the color of the points
  @type pts: list, each element is a 1x3 vector 
  @param pts: the list of points  
  @type size: number
  @param size: the width of the points
  @rtype: rviz visual marker
  @return: the point-list marker
  '''
  marker = createMarker(ns, id, rgb)
  marker.type = Marker.POINTS
  marker.action = Marker.ADD
  marker.pose.orientation.x = 0.0
  marker.pose.orientation.y = 0.0
  marker.pose.orientation.z = 0.0
  marker.pose.orientation.w = 1.0
  marker.scale.x = size
  marker.scale.y = size
  c = 0.1
  step = (1.0-c)/len(pts)
  c = 1.0
  
  for i in xrange(len(pts)):
    marker.points.append(Point(pts[i][0],pts[i][1],pts[i][2]))
    if len(indices) == 0: continue
    
    if i == indices[0]:
      marker.colors.append(ColorRGBA(1, 0, 0, 1))
    elif i == indices[1]:
      marker.colors.append(ColorRGBA(0, 1, 0, 1))
    else:
      marker.colors.append(ColorRGBA(0, 0, c, 1))
    c -= step
    
  return marker


def createConeMarker(ns, id, rgb, finger, cone, planeLength=0.6):
  '''
  Create a cone marker. 
  
  @type ns: string
  @param ns: the marker's namespace in rviz
  @type id: scalar
  @param id: the marker's ID in rviz
  @type rgb: 1x3 vector
  @param rgb: the color of the points
  @type pt: 1x3 vector
  @param pt: a point on the plane
  @type axis: 1x3 vector
  @param axis: the axis that goes along the plane
  @type normal: 1x3 vector
  @param normal: the plane's normal  
  @type length: scalar
  @param length: the length of each side of the rectangle representing the plane
  @rtype: rviz visual marker
  @return: the cone marker
  '''
  coneDirection = [cone.direction[0], cone.direction[1], cone.direction[2], 1]
  rot = tf.transformations.rotation_matrix(cone.angle, finger)
  coneDirection1 = numpy.dot(rot, coneDirection)
  rot = tf.transformations.rotation_matrix(-cone.angle, finger)
  coneDirection2 = numpy.dot(rot, coneDirection)
  
  marker = createMarker(ns, id, rgb)
  marker.type = Marker.LINE_LIST
  marker.action = Marker.ADD
  marker.pose.orientation.x = 0.0
  marker.pose.orientation.y = 0.0
  marker.pose.orientation.z = 0.0
  marker.pose.orientation.w = 1.0
  marker.scale.x = 0.01
  v0 = cone.point + 0.5*planeLength*cone.direction
  v1 = cone.point
  v2 = cone.point + planeLength*coneDirection1[0:3]
  v3 = cone.point + planeLength*coneDirection2[0:3]
  # print "coneDirection1:", coneDirection1
  # print coneDirection2
  marker.points.append(Point(v1[0],v1[1],v1[2]))
  marker.points.append(Point(v0[0],v0[1],v0[2]))
  marker.points.append(Point(v1[0],v1[1],v1[2]))
  marker.points.append(Point(v2[0],v2[1],v2[2]))
  marker.points.append(Point(v1[0],v1[1],v1[2]))
  marker.points.append(Point(v3[0],v3[1],v3[2]))
  return marker
  

def createPlaneMarker(ns, id, rgb, pt, axis, normal, length=0.6):
  '''
  Create a plane marker. 
  
  @type ns: string
  @param ns: the marker's namespace in rviz
  @type id: scalar
  @param id: the marker's ID in rviz
  @type rgb: 1x3 vector
  @param rgb: the color of the points
  @type pt: 1x3 vector
  @param pt: a point on the plane
  @type axis: 1x3 vector
  @param axis: the axis that goes along the plane
  @type normal: 1x3 vector
  @param normal: the plane's normal  
  @type length: scalar
  @param length: the length of each side of the rectangle representing the plane
  @rtype: rviz visual marker
  @return: the plane marker
  '''
  marker = createMarker(ns, id, rgb)
  marker.type = Marker.LINE_STRIP
  marker.action = Marker.ADD
  marker.pose.orientation.x = 0.0
  marker.pose.orientation.y = 0.0
  marker.pose.orientation.z = 0.0
  marker.pose.orientation.w = 1.0
  marker.scale.x = 0.01
  bitangent = numpy.cross(axis, normal)    
  v1 = pt - axis*length - bitangent*length;
  v2 = pt + axis*length - bitangent*length;
  v3 = pt + axis*length + bitangent*length;
  v4 = pt - axis*length + bitangent*length;
  marker.points.append(Point(v1[0],v1[1],v1[2]))
  marker.points.append(Point(v2[0],v2[1],v2[2]))
  marker.points.append(Point(v3[0],v3[1],v3[2]))
  marker.points.append(Point(v4[0],v4[1],v4[2]))
  marker.points.append(Point(v1[0],v1[1],v1[2]))
  return marker


def createSphereMarker(ns, id, rgb, pt, scale=0.03, alpha=1):
  '''
  Create a sphere marker.
  
  @type ns: string
  @param ns: the marker's namespace in rviz
  @type id: scalar
  @param id: the marker's ID in rviz
  @type rgb: 1x3 vector
  @param rgb: the color of the points
  @type pt: 1x3 vector
  @param pt: the centroid of the sphere
  @rtype: rviz visual marker
  @return: the sphere marker       
  '''
  marker = createMarker(ns, id, rgb, alpha)
  marker.type = Marker.SPHERE
  marker.action = Marker.ADD
  marker.pose.position.x = pt[0]
  marker.pose.position.y = pt[1]
  marker.pose.position.z = pt[2]    
  marker.pose.orientation.x = 0.0
  marker.pose.orientation.y = 0.0
  marker.pose.orientation.z = 0.0
  marker.pose.orientation.w = 1.0
  marker.scale.x = scale
  marker.scale.y = scale
  marker.scale.z = scale
  return marker


def createSphereWithCenterMarker(ns, rgb, pt, scale):
  sphere = createSphereMarker(ns, 0, rgb, pt, scale, alpha=0.5)
  center = createSphereMarker(ns, 1, rgb, pt, 0.01, alpha=1)
  markerArray = MarkerArray()
  markerArray.markers.append(sphere)
  markerArray.markers.append(center)
  return markerArray


def createGraspMarker(ns, id, rgb, alpha, center, approach, arrowLength=0.15, scale=0.01, duration=100):
  '''
  Create a grasp marker.
  
  @type ns: string
  @param ns: the marker's namespace in rviz
  @type id: scalar
  @param id: the marker's ID in rviz
  @type rgb: 1x3 vector
  @param rgb: the color of the points
  @type center: 1x3 vector
  @param center: the grasp position 
  @type approach: 1x3 vector
  @param approach: the grasp approach vector 
  '''
  p = center
  q = p - arrowLength*approach
  marker = createMarker(ns, id, rgb, alpha)
  marker.type = Marker.ARROW
  marker.points.append(Point(p[0],p[1],p[2]))
  marker.points.append(Point(q[0],q[1],q[2]))
  marker.scale.x = marker.scale.y = marker.scale.z = scale
  marker.lifetime = rospy.Duration(duration)
  return marker


def createFingersMarker(ns, id, grasp, rgb, alpha, duration=100, approachLength=0.07):
  '''
  Create a 2-fingers marker.   
  '''
  marker = createMarker(ns, id, rgb, alpha)
  marker.type = Marker.LINE_LIST
  
  center = grasp.bottom
  approach_center = center - approachLength*grasp.approach
  left_base = center - 0.5*grasp.width*grasp.binormal
  right_base = center + 0.5*grasp.width*grasp.binormal
  left_end = left_base + approachLength*grasp.approach
  right_end = right_base + approachLength*grasp.approach  
  
  # approach line  
  marker.points.append(Point(center[0], center[1], center[2]))
  marker.points.append(Point(approach_center[0], approach_center[1], approach_center[2]))
  
  # base line
  marker.points.append(Point(left_base[0], left_base[1], left_base[2]))
  marker.points.append(Point(right_base[0], right_base[1], right_base[2]))
  
  # left finger
  marker.points.append(Point(left_base[0], left_base[1], left_base[2]))
  marker.points.append(Point(left_end[0], left_end[1], left_end[2]))
  
  # right finger
  marker.points.append(Point(right_base[0], right_base[1], right_base[2]))
  marker.points.append(Point(right_end[0], right_end[1], right_end[2]))  
  
  marker.scale.x = marker.scale.y = marker.scale.z = 0.008
  marker.lifetime = rospy.Duration(duration)
  
  return marker


def createCubeMarker(ns, id, pos, orient, rgb, alpha=0.5, duration=100):
  '''
  Create a finger marker.   
  '''
  marker = createMarker(ns, id, rgb, alpha)
  marker.type = Marker.CUBE  
  marker.lifetime = rospy.Duration(duration)
  
  marker.pose.position.x = pos[0]
  marker.pose.position.y = pos[1]
  marker.pose.position.z = pos[2]
  
  R = numpy.eye(4)
  R[:3,:3] = orient
  quat = quaternion_from_matrix(R)
  marker.pose.orientation.x = quat[0]
  marker.pose.orientation.y = quat[1]
  marker.pose.orientation.z = quat[2]
  marker.pose.orientation.w = quat[3]
  
  marker.scale.x = 1
  marker.scale.y = 1
  marker.scale.z = 0.01
  
  return marker


def createFinger3D(ns, id, pos, orient, length, width, height, rgba):
  '''Create a marker for a robot hand finger.'''
  
  marker = createCubeMarker(ns, id, pos, orient, rgb=rgba[0:3], alpha=rgba[3])
  
  # these scales are relative to the hand frame (unit: meters)
  marker.scale.x = length # forward direction
  marker.scale.y = width # closing direction
  marker.scale.z = height # vertical direction
  
  return marker
  
  
def createHandBase3D(ns, id, start, end, orient, length, height, rgba):
  '''Create a marker for a robot hand base.'''
  
  center = start + 0.5*(end - start);
  marker = createCubeMarker(ns, id, center, orient, rgb=rgba[0:3], alpha=rgba[3])
  
  # these scales are relative to the hand frame (unit: meters)
  marker.scale.x = length # forward direction
  marker.scale.y = numpy.linalg.norm(end - start) # hand closing direction
  marker.scale.z = height # hand vertical direction
  
  return marker


def createHandMarkers(grasp, handDepth, handHeight, outerDiameter, fingerWidth, id, rgba):
  
  nsBases = 'hand_bases'; nsFingers = 'fingers'
  halfWidth = 0.5 * outerDiameter
  
  leftBottom = grasp.bottom + halfWidth * grasp.binormal
  rightBottom = grasp.bottom - halfWidth * grasp.binormal
  leftTop = leftBottom + handDepth * grasp.approach
  rightTop = rightBottom + handDepth * grasp.approach
  leftCenter = leftBottom + 0.5*(leftTop - leftBottom) - 0.5*fingerWidth*grasp.binormal
  rightCenter = rightBottom + 0.5*(rightTop - rightBottom) + 0.5*fingerWidth*grasp.binormal
  approachCenter = leftBottom + 0.5*(rightBottom - leftBottom) - 0.04*grasp.approach
  
  orient = numpy.eye(3)
  orient[:3,0] = grasp.approach
  orient[:3,1] = grasp.binormal
  orient[:3,2] = grasp.axis
  
  base = createHandBase3D(nsBases, id, leftBottom, rightBottom, orient, length=0.02, height=handHeight, rgba=rgba)
  leftFinger = createFinger3D(nsFingers, id*3, leftCenter, orient, handDepth, fingerWidth, handHeight, rgba)
  rightFinger = createFinger3D(nsFingers, id*3+1, rightCenter, orient, handDepth, fingerWidth, handHeight, rgba)
  approach = createFinger3D(nsFingers, id*3+2, approachCenter, orient, handDepth, fingerWidth, handHeight, rgba)
  
  return [base, leftFinger, rightFinger, approach]


def createGraspsMarkerArray(grasps, color=[], handDepth=0.06, handHeight=0.02, outerDiameter=0.105, fingerWidth=0.01, 
                            rgba=[0,0,0.5,0.5]):
  '''Create a marker array of 3D hands.'''  
  
  markerArray = MarkerArray()
  
  for i,g in enumerate(grasps):
    handMarkers = createHandMarkers(g, handDepth, handHeight, outerDiameter, fingerWidth, i, rgba)
    for m in handMarkers:
      markerArray.markers.append(m)
      
  return markerArray

def createGraspsPosCube(graspSpace, rgba=[0,0,0.5,0.5]):
  marker = createMarker('graspSpace', 100, rgba[:3], rgba[-1])
  marker.type = Marker.CUBE
  marker.lifetime = rospy.Duration(100)

  marker.pose.position.x = (graspSpace[0][0] + graspSpace[0][1])/2.
  marker.pose.position.y = (graspSpace[1][0] + graspSpace[1][1])/2.
  marker.pose.position.z = (graspSpace[2][0] + graspSpace[2][1])/2.

  marker.scale.x = graspSpace[0][1] - graspSpace[0][0]
  marker.scale.y = graspSpace[1][1] - graspSpace[1][0]
  marker.scale.z = graspSpace[2][1] - graspSpace[2][0]

  return marker


def createDeleteMarker(ns, id):
  '''
  Create a transparent marker.
  
  @type ns: string
  @param ns: the marker's namespace in rviz
  @type id: scalar
  @param id: the marker's ID in rviz
  @rtype: rviz visual marker
  @return: the marker with alpha=0 (transparent)       
  '''
  marker = createMarker(ns, id, [0,0,0], 0, action=Marker.DELETEALL)
  marker.scale.x = marker.scale.y = marker.scale.z = 0.01 
  return marker


def createTransparentMarkerArray(ns, n=100):
  '''
  Create a transparent marker.
  
  @type ns: string
  @param ns: the marker's namespace in rviz
  @type id: scalar
  @param id: the marker's ID in rviz
  @rtype: rviz visual marker
  @return: the marker with alpha=0 (transparent)       
  '''
  markerArray = MarkerArray()
  
  for i in xrange(n):
    marker = createMarker(ns, i, [0,0,0], alpha=0.1, action=Marker.ADD)
    marker.scale.x = marker.scale.y = marker.scale.z = 0.01
    markerArray.markers.append(marker)
  
  return markerArray


def createPoseMarker(ns, id, pose, lineLength=0.10, lineWidth=0.01, alpha=1.0):
  '''
  Create a three-perpendicular-lines-marker for a given pose.
  
  The axes of the pose are drawn in the following colors: x-axis: red, y-axis: green, z-axis: blue.
  
  @type ns: string
  @param ns: the marker's namespace in rviz
  @type id: scalar
  @param id: the marker's ID in rviz
  @type pose: 4x4 matrix
  @param pose: the given pose
  @rtype: rviz visual marker
  @return: the marker with alpha=0 (transparent)       
  '''
  marker = createMarker(ns, id, [0,0,0])
  marker.type = Marker.LINE_LIST
  marker.scale.x = lineWidth # width of line segment
  origin = pose[0:3,3]
  x = origin + lineLength*pose[0:3,0]
  y = origin + lineLength*pose[0:3,1]
  z = origin + lineLength*pose[0:3,2]
  marker.points.append(Point(origin[0],origin[1],origin[2]))
  marker.points.append(Point(x[0],x[1],x[2]))
  marker.points.append(Point(origin[0],origin[1],origin[2]))
  marker.points.append(Point(y[0],y[1],y[2]))
  marker.points.append(Point(origin[0],origin[1],origin[2]))
  marker.points.append(Point(z[0],z[1],z[2]))
  marker.colors.append(ColorRGBA(1, 0, 0, alpha))
  marker.colors.append(ColorRGBA(1, 0, 0, alpha))
  marker.colors.append(ColorRGBA(0, 1, 0, alpha))
  marker.colors.append(ColorRGBA(0, 1, 0, alpha))
  marker.colors.append(ColorRGBA(0, 0, 1, alpha))
  marker.colors.append(ColorRGBA(0, 0, 1, alpha))
  return marker
