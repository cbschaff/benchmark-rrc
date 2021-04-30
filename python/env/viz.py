from mp.utils import Transform, get_rotation_between_vecs
from scipy.spatial.transform import Rotation as R
import numpy as np
import pybullet as p


class Marker:
    """
    In case any point(for eg. the goal position) in space is to be
    visualized using a marker.
    """

    def __init__(
        self,
        number_of_goals,
        goal_size=0.015,
        initial_position=[0.18, 0.18, 0.08],
        **kwargs,
    ):
        """
        Import a marker for visualization

        Args:
            number_of_goals (int): the desired number of goals to display
            goal_size (float): how big should this goal be
            initial_position (list of floats): where in xyz space should the
                goal first be displayed
        """
        self._kwargs = kwargs
        color_cycle = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]

        goal_shape_ids = [None] * number_of_goals
        self.goal_ids = [None] * number_of_goals
        self.goal_orientations = [None] * number_of_goals

        # Can use both a block, or a sphere: uncomment accordingly
        for i in range(number_of_goals):
            color = color_cycle[i % len(color_cycle)]
            goal_shape_ids[i] = p.createVisualShape(
                # shapeType=pybullet.GEOM_BOX,
                # halfExtents=[goal_size] * number_of_goals,
                shapeType=p.GEOM_SPHERE,
                radius=goal_size,
                rgbaColor=color,
                **self._kwargs,
            )
            self.goal_ids[i] = p.createMultiBody(
                baseVisualShapeIndex=goal_shape_ids[i],
                basePosition=initial_position,
                baseOrientation=[0, 0, 0, 1],
                **self._kwargs,
            )
            (
                _,
                self.goal_orientations[i],
            ) = p.getBasePositionAndOrientation(
                self.goal_ids[i],
                **self._kwargs,
            )

    def set_state(self, positions):
        """
        Set new positions for the goal markers with the orientation being the
        same as when they were imported.

        Args:
            positions (list of lists):  List of lists with
                x,y,z positions of all goals.
        """
        for goal_id, orientation, position in zip(
            self.goal_ids, self.goal_orientations, positions
        ):
            p.resetBasePositionAndOrientation(
                goal_id,
                position,
                orientation,
                **self._kwargs,
            )


class ObjectMarker:
    def __init__(
        self,
        shape_type,
        position,
        orientation,
        color=(0, 1, 0, 0.5),
        pybullet_client_id=0,
        **kwargs,
    ):
        """
        Create a cube marker for visualization

        Args:
            shape_type: Shape type of the object (e.g. pybullet.GEOM_BOX).
            position: Position (x, y, z)
            orientation: Orientation as quaternion (x, y, z, w)
            kwargs: Keyword arguments that are passed to
                pybullet.createVisualShape.  Use this to specify
                shape-type-specify parameters like the object size.
        """
        self._pybullet_client_id = pybullet_client_id

        self.shape_id = p.createVisualShape(
            shapeType=shape_type,
            rgbaColor=color,
            physicsClientId=self._pybullet_client_id,
            **kwargs,
        )
        self.body_id = p.createMultiBody(
            baseVisualShapeIndex=self.shape_id,
            basePosition=position,
            baseOrientation=orientation,
            physicsClientId=self._pybullet_client_id,
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
            orientation,
            physicsClientId=self._pybullet_client_id,
        )


class CuboidMarker(ObjectMarker):
    """Visualize a Cuboid."""

    def __init__(
        self,
        size,
        position,
        orientation,
        color=(0, 1, 0, 0.5),
        pybullet_client_id=0,
    ):
        """
        Create a cube marker for visualization

        Args:
            size (list): Lengths of the cuboid sides.
            position: Position (x, y, z)
            orientation: Orientation as quaternion (x, y, z, w)
            color: Color of the cube as a tuple (r, b, g, a)
        """
        size = np.asarray(size)
        super().__init__(
            p.GEOM_BOX,
            position,
            orientation,
            color,
            pybullet_client_id,
            halfExtents=size / 2,
        )


class CubeMarker(CuboidMarker):
    """Visualize a cube."""

    def __init__(
        self,
        width,
        position,
        orientation,
        color=(0, 1, 0, 0.5),
        pybullet_client_id=0,
    ):
        """
        Create a cube marker for visualization

        Args:
            width (float): Length of one side of the cube.
            position: Position (x, y, z)
            orientation: Orientation as quaternion (x, y, z, w)
            color: Color of the cube as a tuple (r, b, g, a)
        """
        super().__init__(
            [width] * 3,
            position,
            orientation,
            color,
            pybullet_client_id,
        )


class VisualMarkers:
    '''Visualize spheres on the specified points'''

    def __init__(self):
        self.markers = []

    def add(self, points, radius=0.015, color=None):
        if isinstance(points[0], (int, float)):
            points = [points]
        if color is None:
            color = (0, 1, 1, 0.5)
        for point in points:
            self.markers.append(SphereMaker(radius, point, color=color))

    def remove(self):
        self.markers = []

    def __del__(self):
        for marker in self.markers:
            marker.__del__()


class VisualCubeOrientation:
    '''visualize cube orientation by three cylinder'''

    def __init__(self, cube_position, cube_orientation, cube_halfwidth=0.0325):
        self.markers = []
        self.cube_halfwidth = cube_halfwidth

        color_cycle = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]

        self.z_axis = np.asarray([0, 0, 1])

        const = 1 / np.sqrt(2)
        x_rot = R.from_quat([const, 0, const, 0])
        y_rot = R.from_quat([0, const, const, 0])
        z_rot = R.from_quat([0, 0, 0, 1])

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
        self, radius, length, position, orientation, color=(0, 1, 0, 0.5)
    ):
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


class SphereMaker:
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


class Viz(object):
    def __init__(self):
        self.goal_viz = None
        self.cube_viz = None
        self.initialized = False
        self.markers = []

    def reset(self):
        if self.goal_viz:
            del self.goal_viz
            del self.cube_viz

        self.goal_viz = None
        self.cube_viz = None

        if self.markers:
            for m in self.markers:
                del m
        self.markers = []
        self.initialized = True

    def update_cube_orientation(self, pos, ori, goal_pos, goal_ori):
        if self.goal_viz is None:
            self.goal_viz = VisualCubeOrientation(goal_pos, goal_ori)
        if self.cube_viz is None:
            self.cube_viz = VisualCubeOrientation(pos, ori)
        else:
            self.cube_viz.set_state(pos, ori)

    def update_tip_force_markers(self, obs, tip_wrenches, force):
        R_cube_to_base = Transform(np.zeros(3), obs['object_orientation'])
        cube_force = R_cube_to_base(force)
        self._set_markers([w[:3] for w in tip_wrenches],
                          obs['robot_tip_positions'],
                          cube_force, obs['object_position'])

    def _set_markers(self, forces, tips, cube_force, cube_pos):
        R = 0.005
        L = 0.2
        ms = []
        cube_force = cube_force / np.linalg.norm(cube_force)
        q = get_rotation_between_vecs(np.array([0, 0, 1]), cube_force)
        if not self.markers:
            ms.append(CylinderMarker(R, L, cube_pos + 0.5 * L * cube_force,
                                     q, color=(0, 0, 1, 0.5)))
        else:
            self.markers[0].set_state(cube_pos + 0.5 * L * cube_force, q)
            ms.append(self.markers[0])

        for i, (f, t) in enumerate(zip(forces, tips)):
            f = f / np.linalg.norm(f)
            q = get_rotation_between_vecs(np.array([0, 0, 1]), f)
            if not self.markers:
                ms.append(CylinderMarker(R, L, t + 0.5 * L * f,
                                         q, color=(1, 0, 0, 0.5)))
            else:
                self.markers[i+1].set_state(t + 0.5 * L * f, q)
                ms.append(self.markers[i+1])
        self.markers = ms
