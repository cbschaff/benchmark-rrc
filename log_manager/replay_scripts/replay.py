#!/usr/bin/env python3

import os
import shelve
import argparse
import robot_fingers
import trifinger_simulation
import pybullet as p
import numpy as np
from trifinger_simulation.tasks import move_cube
from trifinger_simulation import camera, visual_objects
import trifinger_object_tracking.py_tricamera_types as tricamera
import trifinger_cameras
from trifinger_cameras.utils import convert_image
from scipy.spatial.transform import Rotation as R
import cv2
import json


def load_data(path):
    data = {}
    try:
        with shelve.open(path) as f:
            for key, val in f.items():
                data[key] = val
    except Exception:
        return {}
    return data


class SphereMarker:
    def __init__(self, radius, position, color=(0, 1, 0, 0.5)):
        """
        Create a sphere marker for visualization

        Args:
            width (float): Length of one side of the cube.
            position: Position (x, y, z)
            orientation: Orientation as quaternion (x, y, z, w)
            color: Color of the cube as a tuple (r, b, g, q)
            """
        self.shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color,
        )
        self.body_id = p.createMultiBody(
            baseVisualShapeIndex=self.shape_id,
            basePosition=position,
            baseOrientation=[0, 0, 0, 1],
        )

    def set_state(self, position):
        """Set pose of the marker.

        Args:
            position: Position (x, y, z)
        """
        orientation = [0, 0, 0, 1]
        p.resetBasePositionAndOrientation(
            self.body_id, position, orientation
        )

    def __del__(self):
        """
        Removes the visual object from the environment
        """
        # At this point it may be that pybullet was already shut down. To avoid
        # an error, only remove the object if the simulation is still running.
        if p.isConnected():
            p.removeBody(self.body_id)


class VisualCubeOrientation:
    '''visualize cube orientation by three cylinder'''
    def __init__(self, cube_position, cube_orientation, cube_halfwidth=0.0325):
        self.markers = []
        self.cube_halfwidth = cube_halfwidth

        color_cycle = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]

        self.z_axis = np.asarray([0,0,1])

        const = 1 / np.sqrt(2)
        x_rot = R.from_quat([const, 0, const, 0])
        y_rot = R.from_quat([0, const, const, 0])
        z_rot = R.from_quat([0,0,0,1])

        assert( np.linalg.norm( x_rot.apply(self.z_axis) - np.asarray([1., 0., 0.]) ) < 0.00000001)
        assert( np.linalg.norm( y_rot.apply(self.z_axis) - np.asarray([0., 1., 0.]) ) < 0.00000001)
        assert( np.linalg.norm( z_rot.apply(self.z_axis) - np.asarray([0., 0., 1.]) ) < 0.00000001)

        self.rotations = [x_rot, y_rot, z_rot]
        cube_rot = R.from_quat(cube_orientation)

        #x: red , y: green, z: blue
        for rot, color in zip(self.rotations, color_cycle):
            rotation = cube_rot * rot
            orientation = rotation.as_quat()
            bias = rotation.apply(self.z_axis) * cube_halfwidth
            self.markers.append(
                CylinderMarker(radius=cube_halfwidth/20,
                               length=cube_halfwidth*2,
                               position=cube_position + bias,
                               orientation=orientation,
                               color=color)
            )

    def set_state(self, position, orientation):
        cube_rot = R.from_quat(orientation)
        for rot, marker in zip(self.rotations, self.markers):
            rotation = cube_rot * rot
            orientation = rotation.as_quat()
            bias = rotation.apply(self.z_axis) * self.cube_halfwidth
            marker.set_state(position=position + bias,
                             orientation=orientation)


class CylinderMarker:
    """Visualize a cylinder."""

    def __init__(
        self, radius, length, position, orientation, color=(0, 1, 0, 0.5)):
        """
        Create a cylinder marker for visualization

        Args:
            radius (float): radius of cylinder.
            length (float): length of cylinder.
            position: Position (x, y, z)
            orientation: Orientation as quaternion (x, y, z, w)
            color: Color of the cube as a tuple (r, b, g, q)
        """

        self.shape_id = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            length=length,
            rgbaColor=color
        )
        self.body_id = p.createMultiBody(
            baseVisualShapeIndex=self.shape_id,
            basePosition=position,
            baseOrientation=orientation
        )

    def set_state(self, position, orientation):
        """Set pose of the marker.

        Args:
            position: Position (x, y, z)
            orientation: Orientation as quaternion (x, y, z, w)
        """
        p.resetBasePositionAndOrientation(
            self.body_id,
            position,
            orientation
        )


class CubeDrawer:
    def __init__(self, logdir):
        calib_files = []
        for name in ("camera60", "camera180", "camera300"):
            calib_files.append(os.path.join(logdir, name + ".yml"))
        self.cube_visualizer = tricamera.CubeVisualizer(calib_files)

    def add_cube(self, images, object_pose):
        cvmats = [trifinger_cameras.camera.cvMat(img) for img in images]
        images = self.cube_visualizer.draw_cube(cvmats, object_pose, False)
        images = [np.array(img) for img in images]

        images = [cv2.putText(
            image,
            "confidence: %.2f" % object_pose.confidence,
            (0, image.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0)
        ) for image in images]
        return images


class VideoRecorder:
    def __init__(self, fps, image_size=(270, 270)):
        self.fps = fps
        self.image_size = image_size
        self.frame_size = None
        self.cameras = camera.TriFingerCameras(image_size=image_size)
        self._add_new_camera()
        self.frames = []

    def _add_new_camera(self):
        self.cameras.cameras.append(
            camera.Camera(
                camera_position=[0.0, 0.0, 0.24],
                camera_orientation=p.getQuaternionFromEuler((0, np.pi, 0))
            )
        )

    def get_views(self):
        images = [cam.get_image() for cam in self.cameras.cameras]
        three_views = np.concatenate((*images,), axis=1)
        return three_views

    def capture_frame(self):
        three_views = self.get_views()
        self.add_frame(three_views)
        return three_views

    def add_frame(self, frame):
        if self.frame_size is None:
            self.frame_size = frame.shape[:2]
        assert frame.shape[:2] == self.frame_size
        self.frames.append(frame)

    def save_video(self, filepath):
        out = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'XVID'),
                              self.fps, (self.frame_size[1], self.frame_size[0]))
        for frame in self.frames:
            out.write(frame)
        out.release()


def get_synced_log_data(logdir, goal, difficulty):
    log = robot_fingers.TriFingerPlatformWithObjectLog(
        os.path.join(logdir, "robot_data.dat"),
        os.path.join(logdir, "camera_data.dat"),
    )
    log_camera = tricamera.LogReader(os.path.join(logdir, "camera_data.dat"))
    stamps = log_camera.timestamps

    obs = {'robot': [], 'cube': [], 'images': [], 't': [], 'desired_action': [],
           'stamp': [], 'acc_reward': []}
    ind = 0
    acc_reward = 0.0
    for t in range(log.get_first_timeindex(), log.get_last_timeindex()):
        camera_observation = log.get_camera_observation(t)
        acc_reward -= move_cube.evaluate_state(
            move_cube.Pose(**goal), camera_observation.filtered_object_pose,
            difficulty
        )
        if 1000 * log.get_timestamp_ms(t) >= stamps[ind]:
            robot_observation = log.get_robot_observation(t)
            obs['robot'].append(robot_observation)
            obs['cube'].append(camera_observation.filtered_object_pose)
            obs['images'].append([convert_image(camera.image)
                                  for camera in camera_observation.cameras])
            obs['desired_action'].append(log.get_desired_action(t))
            obs['acc_reward'].append(acc_reward)
            obs['t'].append(t)
            obs['stamp'].append(log.get_timestamp_ms(t))
            ind += 1
    return obs


def get_goal(logdir):
    filename = os.path.join(logdir, 'goal.json')
    with open(filename, 'r') as f:
        goal = json.load(f)
    return goal['goal'], goal['difficulty']


def add_text(frame, text, position, **kwargs):
    frame = cv2.putText(frame, text, position,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 0),
                        thickness=2, lineType=cv2.LINE_AA, **kwargs)
    return frame

def vstack_frames(frames):
    max_width = max([frame.shape[1] for frame in frames])
    padded_frames = []
    for frame in frames:
        padded_frames.append(
            np.pad(frame, [(0, 0), (0, max_width - frame.shape[1]), (0, 0)], mode='constant')
        )
    return np.concatenate(padded_frames, axis=0)

def main(logdir, video_path):
    goal, difficulty = get_goal(logdir)
    data = get_synced_log_data(logdir, goal, difficulty)
    fps = len(data['t']) / (data['stamp'][-1] - data['stamp'][0])
    video_recorder = VideoRecorder(fps)
    cube_drawer = CubeDrawer(logdir)

    initial_object_pose = move_cube.Pose(data['cube'][0].position,
                                         data['cube'][0].orientation)
    platform = trifinger_simulation.TriFingerPlatform(
        visualization=True,
        initial_object_pose=initial_object_pose,
    )
    markers = []
    marker_cube_ori = VisualCubeOrientation(data['cube'][0].position,
                                            data['cube'][0].orientation)
    marker_goal_ori = VisualCubeOrientation(goal['position'], goal['orientation'])

    visual_objects.CubeMarker(
        width=0.065,
        position=goal['position'],
        orientation=goal['orientation']
    )

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0,0,0])

    for i, t in enumerate(data['t']):
        platform.simfinger.reset_finger_positions_and_velocities(data['desired_action'][i].position)
        platform.cube.set_state(data['cube'][i].position, data['cube'][i].orientation)
        marker_cube_ori.set_state(data['cube'][i].position, data['cube'][i].orientation)
        frame_desired = video_recorder.get_views()
        frame_desired = cv2.cvtColor(frame_desired, cv2.COLOR_RGB2BGR)
        platform.simfinger.reset_finger_positions_and_velocities(data['robot'][i].position)
        frame_observed = video_recorder.get_views()
        frame_observed = cv2.cvtColor(frame_observed, cv2.COLOR_RGB2BGR)
        frame_real = np.concatenate(data['images'][i], axis=1)
        frame_real_cube = np.concatenate(cube_drawer.add_cube(data['images'][i],
                                                              data['cube'][i]),
                                         axis=1)

        frame = vstack_frames((frame_desired, frame_observed, frame_real, frame_real_cube))
        # frame = np.concatenate((frame_desired, frame_observed,
        #                         frame_real, frame_real_cube), axis=0)
        # add text
        frame = add_text(frame, text="step: {:06d}".format(t), position=(10, 40))
        frame = add_text(frame, text="acc reward: {:.3f}".format(data["acc_reward"][i]), position=(10, 70))
        frame = add_text(
            frame,
            text="tip force {}".format(
                np.array2string(data["robot"][i].tip_force, precision=3),
            ),
            position=(10, 100),
        )
        video_recorder.add_frame(frame)
    video_recorder.save_video(video_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", help="path to the log directory")
    parser.add_argument("video_path", help="video file to save (.avi file)")
    args = parser.parse_args()
    main(args.logdir, args.video_path)
