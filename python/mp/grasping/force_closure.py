import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError
from mp.utils import Transform, get_rotation_between_vecs


class FrictionModel:
    def wrench_basis(self):
        pass

    def is_valid(self, wrench):
        pass

    def approximate_cone(self, contacts):
        pass

    def get_forces_from_approx(self, forces):
        pass


class NoFriction(FrictionModel):
    def __init__(self):
        self.basis = np.array([0, 0, 1, 0, 0, 0])[:, None]

    def wrench_basis(self):
        return self.basis

    def is_valid(self, wrench):
        return wrench >= 0

    def approximate_cone(self, contacts):
        return contacts, self

    def get_forces_from_approx(self, forces):
        return forces


class CoulombFriction(FrictionModel):
    def __init__(self, mu):
        self.mu = mu
        self.basis = np.eye(6)[:, :3]
        self.cone_corners, self.corner_transforms = self._get_cone_corners()

    def wrench_basis(self):
        return self.basis

    def is_valid(self, wrench):
        return np.linalg.norm(wrench[:2]) <= wrench[2]

    def _get_cone_corners(self):
        # approximate cone with an inscribed square
        contact_normal = np.array([0, 0, 1])
        fac = self.mu * np.sqrt(2) / 2
        corners = []
        transforms = []
        for i in [-1, +1]:
            for j in [-1, +1]:
                corner = np.array([i * fac, j * fac, 1])
                corner /= np.linalg.norm(corner)
                q = get_rotation_between_vecs(contact_normal, corner)
                corners.append(corner)
                transforms.append(Transform(pos=np.zeros(3), ori=q))
        return corners, transforms

    def approximate_cone(self, contacts):
        """
        Returns a set of contacts under a simpler friction model
        which approximate the friction cone of contacts.
        """
        new_contacts = []
        for c in contacts:
            new_contacts.extend([c] + [c(T) for T in self.corner_transforms])
        return new_contacts, NoFriction()

    def get_forces_from_approx(self, forces):
        n = len(forces) // 5
        contact_normal = np.array([0, 0, 1])[None]
        contact_forces = []
        assert np.all(forces >= 0)
        for i in range(n):
            force = contact_normal * forces[5 * i]
            for j, c in enumerate(self.corner_transforms):
                f = c(contact_normal * forces[5 * i + j + 1])
                force += f
            contact_forces.append(force[0])
        return contact_forces


class CuboidForceClosureTest(object):
    def __init__(self, halfsize, friction_model):
        self.halfsize = halfsize
        self.friction = friction_model

    def _compute_grasp_matrix(self, contacts, friction=None):
        if friction is None:
            friction = self.friction
        return np.concatenate([c.adjoint().dot(friction.wrench_basis())
                               for c in contacts], axis=1)

    def contact_from_tip_position(self, pos):
        """
        Compute contact frame from tip positions in the cube
        center of mass frame.
        """
        outside = np.abs(pos) >= self.halfsize - 1e-5
        sign = np.sign(pos)
        contact = pos.copy()
        normal = np.zeros(3)

        if np.sum(outside) == 0:
            outside[:] = True
        dist = np.minimum(np.abs(pos - self.halfsize),
                          np.abs(pos + self.halfsize))
        axes = np.argsort(dist)
        ax = [ax for ax in axes if outside[ax]][0]
        contact[ax] = sign[ax] * self.halfsize[ax]
        normal[ax] = -sign[ax]
        q = get_rotation_between_vecs(np.array([0, 0, 1]), normal)
        return Transform(pos=contact, ori=q)

    def force_closure_test(self, T_cube_to_base, tips_base_frame):
        contacts = [self.contact_from_tip_position(tip)
                    for tip in T_cube_to_base.inverse()(tips_base_frame)]
        new_contacts, friction = self.friction.approximate_cone(contacts)
        G = self._compute_grasp_matrix(new_contacts, friction)
        try:
            hull = Delaunay(G.T)
        except QhullError:
            return False
        return hull.find_simplex(np.zeros((6))) >= 0
