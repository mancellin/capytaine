#!/usr/bin/env python
# coding: utf-8
"""Floating bodies to be used in radiation-diffraction problems."""
# Copyright (C) 2017-2019 Matthieu Ancellin
# See LICENSE file at <https://github.com/mancellin/capytaine>

import logging
import copy
from itertools import chain, accumulate, product, zip_longest

import numpy as np
import xarray as xr

from capytaine.meshes.geometry import Abstract3DObject, Plane, inplace_transformation
from capytaine.meshes.meshes import Mesh
from capytaine.meshes.symmetric import build_regular_array_of_meshes
from capytaine.meshes.collections import CollectionOfMeshes

LOG = logging.getLogger(__name__)

TRANSLATION_DOFS_DIRECTIONS = {"surge": (1, 0, 0), "sway": (0, 1, 0), "heave": (0, 0, 1)}
ROTATION_DOFS_AXIS = {"roll": (1, 0, 0), "pitch": (0, 1, 0), "yaw": (0, 0, 1)}


class FloatingBody(Abstract3DObject):
    """A floating body described as a mesh and some degrees of freedom.

    The mesh structure is stored as a Mesh from capytaine.mesh.mesh or a
    CollectionOfMeshes from capytaine.mesh.meshes_collection.

    The degrees of freedom (dofs) are stored as a dict associating a name to
    a complex-valued array of shape (nb_faces, 3). To each face of the body
    (as indexed in the mesh) corresponds a complex-valued 3d vector, which
    defines the displacement of the center of the face in frequency domain.

    Parameters
    ----------
    mesh : Mesh or CollectionOfMeshes, optional
        the mesh describing the geometry of the floating body.
        If none is given, a empty one is created.
    dofs : dict, optional
        the degrees of freedom of the body.
        If none is given, a empty dictionary is initialized.
    name : str, optional
        a name for the body.
        If none is given, the one of the mesh is used.
    """

    def __init__(self, mesh=None, dofs=None, name=None):
        if mesh is None:
            mesh = Mesh(name="dummy_mesh")

        if dofs is None:
            dofs = {}

        if name is None:
            name = mesh.name

        assert isinstance(mesh, Mesh) or isinstance(mesh, CollectionOfMeshes)
        self.mesh = mesh
        self.full_body = None
        self.dofs = dofs
        self.name = name

        if self.mesh.nb_vertices == 0 or self.mesh.nb_faces == 0:
            LOG.warning(f"New floating body (with empty mesh!): {self.name}.")
        else:
            self.mesh.heal_mesh()
            LOG.info(f"New floating body: {self.name}.")

    @staticmethod
    def from_file(filename: str, file_format=None, name=None) -> 'FloatingBody':
        """Create a FloatingBody from a mesh file using meshmagick."""
        from capytaine.io.mesh_loaders import load_mesh
        if name is None:
            name = filename
        mesh = load_mesh(filename, file_format, name=f"{name}_mesh")
        return FloatingBody(mesh, name=name)

    def __lt__(self, other: 'FloatingBody') -> bool:
        """Arbitrary order. The point is to sort together the problems involving the same body."""
        return self.name < other.name

    ##########
    #  Dofs  #
    ##########

    @property
    def nb_dofs(self) -> int:
        """Number of degrees of freedom."""
        return len(self.dofs)

    def add_translation_dof(self, direction=None, name=None, amplitude=1.0) -> None:
        """Add a new translation dof (in place).
        If no direction is given, the code tries to infer it from the name.

        Parameters
        ----------
        direction : array of shape (3,), optional
            the direction of the translation
        name : str, optional
            a name for the degree of freedom
        amplitude : float, optional
            amplitude of the dof (default: 1.0 m/s)
        """
        if direction is None:
            if name is not None and name.lower() in TRANSLATION_DOFS_DIRECTIONS:
                direction = TRANSLATION_DOFS_DIRECTIONS[name.lower()]
            else:
                raise ValueError("A direction needs to be specified for the dof.")

        if name is None:
            name = f"dof_{self.nb_dofs}_translation"

        direction = np.asarray(direction)
        assert direction.shape == (3,)

        motion = np.empty((self.mesh.nb_faces, 3))
        motion[:, :] = direction
        self.dofs[name] = amplitude * motion

    def add_rotation_dof(self, axis=None, name=None, amplitude=1.0) -> None:
        """Add a new rotation dof (in place).
        If no axis is given, the code tries to infer it from the name.

        Parameters
        ----------
        axis: Axis, optional
            the axis of the rotation
        name : str, optional
            a name for the degree of freedom
        amplitude : float, optional
            amplitude of the dof (default: 1.0)
        """
        if axis is None:
            if name is not None and name.lower() in ROTATION_DOFS_AXIS:
                axis_direction = ROTATION_DOFS_AXIS[name.lower()]
                for point_attr in ('rotation_center', 'center_of_mass', 'geometric_center'):
                    if hasattr(self, point_attr):
                        axis_point = getattr(self, point_attr)
                        LOG.info(f"The rotation dof {name} has been initialized around the point: "
                                 f"{self.name}.{point_attr} = {getattr(self, point_attr)}")
                        break
                else:
                    axis_point = np.array([0, 0, 0])
                    LOG.warning(f"The rotation dof {name} has been initialized "
                                f"around the origin of the domain (0, 0, 0).")
            else:
                raise ValueError("A direction needs to be specified for the dof.")
        else:
            axis_point = axis.point
            axis_direction = axis.vector

        if name is None:
            name = f"dof_{self.nb_dofs}_rotation"

        if self.mesh.nb_faces == 0:
            self.dofs[name] = np.empty((self.mesh.nb_faces, 3))
        else:
            motion = np.cross(axis_point - self.mesh.faces_centers, axis_direction)
            self.dofs[name] = amplitude * motion

    def add_all_rigid_body_dofs(self) -> None:
        """Add the six degrees of freedom of rigid bodies (in place)."""
        self.add_translation_dof(name="Surge")
        self.add_translation_dof(name="Sway")
        self.add_translation_dof(name="Heave")
        self.add_rotation_dof(name="Roll")
        self.add_rotation_dof(name="Pitch")
        self.add_rotation_dof(name="Yaw")

    @inplace_transformation
    def keep_only_dofs(self, dofs):
        for dof in list(self.dofs.keys()):
            if dof not in dofs:
                del self.dofs[dof]

        if hasattr(self, 'mass'):
            self.mass = self.mass.sel(radiating_dof=dofs, influenced_dof=dofs)
        if hasattr(self, 'hydrostatic_stiffness'):
            self.hydrostatic_stiffness = self.hydrostatic_stiffness.sel(radiating_dof=dofs, influenced_dof=dofs)

        return self

    def add_dofs_labels_to_vector(self, vector):
        """Helper function turning a bare vector into a vector labelled by the name of the dofs of the body,
        to be used for instance for the computation of RAO."""
        return xr.DataArray(data=np.asarray(vector), dims=['influenced_dof'],
                            coords={'influenced_dof': list(self.dofs)},
                            )

    def add_dofs_labels_to_matrix(self, matrix):
        """Helper function turning a bare matrix into a matrix labelled by the name of the dofs of the body,
        to be used for instance for the computation of RAO."""
        return xr.DataArray(data=np.asarray(matrix), dims=['influenced_dof', 'radiating_dof'],
                            coords={'influenced_dof': list(self.dofs), 'radiating_dof': list(self.dofs)},
                            )

    ###################
    # Transformations #
    ###################

    def __add__(self, body_to_add: 'FloatingBody') -> 'FloatingBody':
        return self.join_bodies(body_to_add)

    def join_bodies(*bodies, name=None) -> 'FloatingBody':
        if name is None:
            name = "+".join(body.name for body in bodies)
        meshes = CollectionOfMeshes([body.mesh for body in bodies], name=f"{name}_mesh")
        dofs = FloatingBody.combine_dofs(bodies)
        return FloatingBody(mesh=meshes, dofs=dofs, name=name)

    @staticmethod
    def combine_dofs(bodies) -> dict:
        """Combine the degrees of freedom of several bodies."""
        dofs = {}
        cum_nb_faces = accumulate(chain([0], (body.mesh.nb_faces for body in bodies)))
        total_nb_faces = sum(body.mesh.nb_faces for body in bodies)
        for body, nbf in zip(bodies, cum_nb_faces):
            # nbf is the cumulative number of faces of the previous subbodies,
            # that is the offset of the indices of the faces of the current body.
            for name, dof in body.dofs.items():
                new_dof = np.zeros((total_nb_faces, 3))
                new_dof[nbf:nbf+len(dof), :] = dof
                if '__' not in name:
                    new_dof_name = '__'.join([body.name, name])
                else:
                    # The body is probably a combination of bodies already.
                    # So for the associativity of the + operation,
                    # it is better to keep the same name.
                    new_dof_name = name
                dofs[new_dof_name] = new_dof
        return dofs

    def copy(self, name=None) -> 'FloatingBody':
        """Return a deep copy of the body.

        Parameters
        ----------
        name : str, optional
            a name for the new copy
        """
        new_body = copy.deepcopy(self)
        if name is None:
            new_body.name = f"copy_of_{self.name}"
            LOG.debug(f"Copy {self.name}.")
        else:
            new_body.name = name
            LOG.debug(f"Copy {self.name} under the name {name}.")
        return new_body

    def assemble_regular_array(self, distance, nb_bodies):
        """Create an regular array of identical bodies.

        Parameters
        ----------
        distance : float
            Center-to-center distance between objects in the array
        nb_bodies : couple of ints
            Number of objects in the x and y directions.

        Returns
        -------
        FloatingBody
        """
        array_mesh = build_regular_array_of_meshes(self.mesh, distance, nb_bodies)
        total_nb_faces = array_mesh.nb_faces
        array_dofs = {}
        for dof_name, dof in self.dofs.items():
            for i, j in product(range(nb_bodies[0]), range(nb_bodies[1])):
                shift_nb_faces = (j*nb_bodies[0] + i) * self.mesh.nb_faces
                new_dof = np.zeros((total_nb_faces, 3))
                new_dof[shift_nb_faces:shift_nb_faces+len(dof), :] = dof
                array_dofs[f'{i}_{j}__{dof_name}'] = new_dof
        return FloatingBody(mesh=array_mesh, dofs=array_dofs, name=f"array_of_{self.name}")

    def extract_faces(self, id_faces_to_extract, return_index=False):
        """Create a new FloatingBody by extracting some faces from the mesh.
        The dofs evolve accordingly.
        """
        if isinstance(self.mesh, CollectionOfMeshes):
            raise NotImplementedError  # TODO

        if return_index:
            new_mesh, id_v = Mesh.extract_faces(self.mesh, id_faces_to_extract, return_index)
        else:
            new_mesh = Mesh.extract_faces(self.mesh, id_faces_to_extract, return_index)
        new_body = FloatingBody(new_mesh)
        LOG.info(f"Extract floating body from {self.name}.")

        new_body.dofs = {}
        for name, dof in self.dofs.items():
            new_body.dofs[name] = dof[id_faces_to_extract, :]

        if return_index:
            return new_body, id_v
        else:
            return new_body

    def sliced_by_plane(self, plane):
        return FloatingBody(mesh=self.mesh.sliced_by_plane(plane), dofs=self.dofs, name=self.name)

    def minced(self, nb_slices=(8, 8, 4)):
        """Experimental method decomposing the mesh as a hierarchical structure.

        Parameters
        ----------
        nb_slices: Tuple[int, int, int]
            The number of slices in each of the x, y and z directions.
            Only powers of 2 are supported at the moment.

        Returns
        -------
        FloatingBody
        """
        minced_body = self.copy()

        # Extreme points of the mesh in each directions.
        x_min, x_max, y_min, y_max, z_min, z_max = self.mesh.axis_aligned_bbox
        sizes = [(x_min, x_max), (y_min, y_max), (z_min, z_max)]

        directions = [np.array(d) for d in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]]

        def _slice_positions_at_depth(i):
            """Helper function.

            Returns a list of floats as follows:
            i=1 -> [1/2]
            i=2 -> [1/4, 3/4]
            i=3 -> [1/8, 3/8, 5/8, 7/8]
                   ...
            """
            denominator = 2**i
            return [numerator/denominator for numerator in range(1, denominator, 2)]

        # GENERATE ALL THE PLANES THAT WILL BE USED TO MINCE THE MESH
        planes = []
        for direction, nb_slices_in_dir, (min_coord, max_coord) in zip(directions, nb_slices, sizes):
            planes_in_dir = []

            depth_of_treelike_structure = int(np.log2(nb_slices_in_dir))
            for i_depth in range(1, depth_of_treelike_structure+1):
                planes_in_dir_at_depth = []
                for relative_position in _slice_positions_at_depth(i_depth):
                    slice_position = (min_coord + relative_position*(max_coord-min_coord))*direction
                    plane = Plane(normal=direction, point=slice_position)
                    planes_in_dir_at_depth.append(plane)
                planes_in_dir.append(planes_in_dir_at_depth)
            planes.append(planes_in_dir)

        # SLICE THE MESH
        intermingled_x_y_z = chain.from_iterable(zip_longest(*planes))
        for planes in intermingled_x_y_z:
            if planes is not None:
                for plane in planes:
                    minced_body = minced_body.sliced_by_plane(plane)
        return minced_body

    @inplace_transformation
    def mirror(self, plane):
        self.mesh.mirror(plane)
        for dof in self.dofs:
            self.dofs[dof] -= 2 * np.outer(np.dot(self.dofs[dof], plane.normal), plane.normal)
        for point_attr in ('geometric_center', 'rotation_center', 'center_of_mass'):
            if point_attr in self.__dict__:
                self.__dict__[point_attr] -= 2 * (np.dot(self.__dict__[point_attr], plane.normal) - plane.c) * plane.normal
        return self

    @inplace_transformation
    def translate(self, *args):
        self.mesh.translate(*args)
        for point_attr in ('geometric_center', 'rotation_center', 'center_of_mass'):
            if point_attr in self.__dict__:
                self.__dict__[point_attr] += args[0]
        return self

    @inplace_transformation
    def rotate(self, axis, angle):
        matrix = axis.rotation_matrix(angle)
        self.mesh.rotate(axis, angle)
        for point_attr in ('geometric_center', 'rotation_center', 'center_of_mass'):
            if point_attr in self.__dict__:
                self.__dict__[point_attr] = matrix @ self.__dict__[point_attr]
        for dof in self.dofs:
            self.dofs[dof] = (matrix @ self.dofs[dof].T).T
        return self

    @inplace_transformation
    def clip(self, plane):
        # Keep of copy of the full mesh
        if self.full_body is None:
            self.full_body = self.copy()

        # Clip mesh
        LOG.info(f"Clipping {self.name} with respect to {plane}")
        self.mesh.clip(plane)

        # Clip dofs
        ids = self.mesh._clipping_data['faces_ids']
        for dof in self.dofs:
            if len(ids) > 0:
                self.dofs[dof] = self.dofs[dof][ids]
            else:
                self.dofs[dof] = np.empty((0, 3))
        return self

    def clipped(self, plane, **kwargs):
        # Same API as for the other transformations
        return self.clip(plane, inplace=False, **kwargs)

    @inplace_transformation
    def keep_immersed_part(self, free_surface=0.0, sea_bottom=-np.infty):
        """Remove the parts of the mesh above the sea bottom and below the free surface."""
        self.clip(Plane(normal=(0, 0, 1), point=(0, 0, free_surface)))
        if sea_bottom > -np.infty:
            self.clip(Plane(normal=(0, 0, -1), point=(0, 0, sea_bottom)))
        return self

    #############
    #  Display  #
    #############

    def __str__(self):
        return self.name

    def __repr__(self):
        return (f"{self.__class__.__name__}(mesh={self.mesh.name}, "
                f"dofs={{{', '.join(self.dofs.keys())}}}, name={self.name})")

    def show(self, **kwargs):
        from capytaine.ui.vtk.body_viewer import FloatingBodyViewer
        viewer = FloatingBodyViewer()
        viewer.add_body(self, **kwargs)
        viewer.show()
        viewer.finalize()

    def show_matplotlib(self, *args, **kwargs):
        return self.mesh.show_matplotlib(*args, **kwargs)

    def animate(self, motion, *args, **kwargs):
        """Display a motion as a 3D animation.

        Parameters
        ==========
        motion: dict or pd.Series or str
            A dict or series mapping the name of the dofs to its amplitude.
            If a single string is passed, it is assumed to be the name of a dof
            and this dof with a unit amplitude will be displayed.
        """
        from capytaine.ui.vtk.animation import Animation
        if isinstance(motion, str):
            motion = {motion: 1.0}
        elif isinstance(motion, xr.DataArray):
            motion = {k: motion.sel(radiating_dof=k).data for k in motion.coords["radiating_dof"].data}
        animation = Animation(*args, **kwargs)
        animation._add_actor(self.mesh.merged(), faces_motion=sum(motion[dof_name] * dof for dof_name, dof in self.dofs.items() if dof_name in motion))
        return animation

