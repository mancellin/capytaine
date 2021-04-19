#!/usr/bin/env python
# coding: utf-8
"""Generate meshes of cylinders and disks"""
# Copyright (C) 2017-2019 Matthieu Ancellin
# See LICENSE file at <https://github.com/mancellin/capytaine>

import logging
from itertools import product

import numpy as np

from capytaine.meshes.geometry import xOy_Plane, xOz_Plane, yOz_Plane, e_x, e_z, Oz_axis
from capytaine.meshes.meshes import Mesh
from capytaine.meshes.collections import CollectionOfMeshes
from capytaine.meshes.symmetric import TranslationalSymmetricMesh, AxialSymmetricMesh, ReflectionSymmetricMesh
from capytaine.bodies.bodies import FloatingBody

LOG = logging.getLogger(__name__)


##########
#  Disk  #
##########

class Disk(FloatingBody):
    """(One-sided) disk.

    Parameters
    ----------
    radius : float, optional
        radius of the disk
    resolution : 2-ple of int, optional
        number of panels along a radius and around the disk
    center : 3-ple or array of shape (3,), optional
        position of the geometric center of the disk
    normal: 3-ple of floats, optional
        normal vector, default: along x axis
    axial_symmetry : bool, optional
        if True, use the axial symmetry to speed up the computations
    reflection_symmetry : bool, optional
        if True, use the reflection symmetry to speed up the computations
    name : str, optional
        a string naming the floating body
    """

    def __init__(self, radius=1.0, resolution=(3, 5),
                 center=(0, 0, 0), normal=(1, 0, 0),
                 reflection_symmetry=False, axial_symmetry=False,
                 name=None):
        assert radius > 0, "Radius of the disk mesh should be given as a positive value."

        assert len(resolution) == 2, "Resolution of a disk should be given as a couple a values."
        assert all([h > 0 for h in resolution]), "Resolution of the disk mesh should be given as positive values."
        assert all([i == int(i) for i in resolution]), "Resolution of a disk should be given as integer values."

        assert len(center) == 3, "Position of the center of a disk should be given a 3-ple of values."

        self.radius = float(radius)
        self.geometric_center = np.asarray(center, dtype=float)
        nr, ntheta = resolution

        if reflection_symmetry and ntheta % 2 == 1:
            raise ValueError("To use the reflection symmetry of the mesh, "
                             "it should have an even number of panels in this direction.")

        if reflection_symmetry and axial_symmetry:
            raise NotImplementedError("Disk generators with both symmetries have not been implemented.")

        if name is None:
            name = f"disk_{next(Mesh._ids)}"

        LOG.debug(f"New disk body of radius {radius} and resolution ({nr}, {ntheta}), named {name}.")

        if reflection_symmetry:
            half_mesh = Disk.generate_disk_mesh(radius=self.radius, theta_max=np.pi/2,
                                                nr=nr, ntheta=ntheta//2,
                                                name=f"half_of_{name}_mesh")
            mesh = ReflectionSymmetricMesh(half_mesh, plane=xOz_Plane, name=f"{name}_mesh")

        elif axial_symmetry:
            mesh_slice = Disk.generate_disk_mesh(radius=self.radius, theta_max=np.pi/ntheta,
                                                 nr=nr, ntheta=1,
                                                 name=f"slice_of_{name}_mesh")
            mesh_slice.rotate_around_center_to_align_vectors((0, 0, 0), e_x, e_z)  # Convoluted way to avoid a warning message in AxialSymmetry...
            mesh = AxialSymmetricMesh(mesh_slice, axis=Oz_axis, nb_repetitions=ntheta - 1, name=f"{name}_mesh")
            mesh.rotate_around_center_to_align_vectors((0, 0, 0), e_z, e_x)

        else:
            mesh = Disk.generate_disk_mesh(radius=self.radius, nr=nr, ntheta=ntheta, name=f"{name}_mesh")

        mesh.rotate_around_center_to_align_vectors((0, 0, 0), mesh.faces_normals[0], normal)
        mesh.translate(center)
        FloatingBody.__init__(self, mesh=mesh, name=name)

    @staticmethod
    def generate_disk_mesh(radius=1.0, theta_max=np.pi,
                           nr=2, ntheta=4,
                           center=(0, 0, 0), normal=(1, 0, 0),
                           name=None) -> Mesh:
        theta_range = np.linspace(0, 2*theta_max, ntheta+1)
        r_range = np.linspace(0.0, radius, nr+1)

        nodes = np.zeros(((ntheta+1)*(nr+1), 3), dtype=float)
        for i, (r, t) in enumerate(product(r_range, theta_range)):
            y = +r * np.sin(t)
            z = -r * np.cos(t)
            nodes[i, :] = (0, y, z)

        panels = np.zeros((ntheta*nr, 4), dtype=int)

        for k, (i, j) in enumerate(product(range(0, nr), range(0, ntheta))):
            panels[k, :] = (
                j+i*(ntheta+1),
                j+1+i*(ntheta+1),
                j+1+(i+1)*(ntheta+1),
                j+(i+1)*(ntheta+1)
            )

        mesh = Mesh(nodes, panels, name=name)
        mesh.merge_duplicates()
        mesh.heal_triangles()
        mesh.rotate_around_center_to_align_vectors((0, 0, 0), mesh.faces_normals[0], normal)
        mesh.translate(center)
        return mesh


##############
#  Cylinder  #
##############

class HorizontalCylinder(FloatingBody):
    """Horizontal cylinder

    Parameters
    ----------
    length : float, optional
        length of the cylinder
    radius : float, optional
        radius of the cylinder
    center : 3-ple or array of shape (3,), optional
        position of the geometric center of the cylinder
    nx : int, optional
        number of circular slices
    ntheta : int, optional
        number of panels along a circular slice of the cylinder
    nr : int, optional
        number of panels along a radius on the extremities of the cylinder
    reflection_symmetry : bool, optional
        if True, returns a ReflectionSymmetricMesh
    translation_symmetry : bool, optional
        if True, uses a TranslationalSymmetricMesh internally for the main part of the cylinder
    name : str, optional
        a string naming the floating body
    """

    def __init__(self, length=10.0, radius=1.0, center=(0, 0, 0),
                 nx=10, ntheta=10, nr=2,
                 reflection_symmetry=True, translation_symmetry=False,
                 clever=None,
                 name=None):
        self.length = length
        self.radius = radius
        self.geometric_center = np.asarray(center, dtype=float)

        if name is None:
            name = f"cylinder_{next(Mesh._ids)}"

        ntheta = 2*(ntheta//2)  # Temporary fix to avoid mismatch in mesh
        # When symmetries are used, one needs an even number of panels.
        # TODO: When symmetries are not used, implement the odd case.

        if clever is not None:
            LOG.warning("Deprecation warning: `clever` argument for VerticalCylinder is deprecated."
                        "Use `reflection_symmetry` and/or `translation_symmetry` instead.")

        open_cylinder = self._generate_open_cylinder_mesh(nx, ntheta,
                                                          reflection_symmetry=reflection_symmetry,
                                                          translation_symmetry=translation_symmetry,
                                                          name=f"body_of_{name}")

        if nr == 0:  # No sides
            mesh = open_cylinder

        else:  # Sides
            side = Disk(radius=radius, center=(-np.array([length/2, 0, 0])), normal=(-1, 0, 0),
                        reflection_symmetry=reflection_symmetry,
                        resolution=(nr, ntheta), name=f"side_of_{name}").mesh

            other_side = side.copy(name=f"other_side_of_{name}_mesh")
            other_side.mirror(yOz_Plane)

            if reflection_symmetry:  # Knit the sides into the symmetric representation of the open cylinder
                half_sides = CollectionOfMeshes((side.half, other_side.half), name="half_sides_of_{name}_mesh")
                half_mesh = CollectionOfMeshes((open_cylinder.half, half_sides), name="half_{name}_mesh")
                mesh = ReflectionSymmetricMesh(half_mesh, plane=xOz_Plane, name=f"{name}_mesh")
            else:
                sides = CollectionOfMeshes((side, other_side), name="sides_of_cylinder_{name}_mesh")
                mesh = CollectionOfMeshes((open_cylinder, sides), name=f"{name}_mesh")

        if not reflection_symmetry and not translation_symmetry:
            mesh = mesh.merged()

        mesh.heal_mesh()

        mesh.translate(self.geometric_center)
        mesh.name = f"{name}_mesh"

        FloatingBody.__init__(self, mesh=mesh, name=name)

    def _generate_open_cylinder_mesh(self, nx, ntheta, reflection_symmetry, translation_symmetry, name=None):
        """Open horizontal cylinder using the symmetries (translation and reflection) to speed up the computations"""
        theta_max = np.pi
        theta = np.linspace(0, theta_max, ntheta//2+1)
        X = np.array([0, self.length/nx])

        # Nodes
        nodes = np.zeros(((ntheta//2+1)*2, 3), dtype=float)

        for i, (t, x) in enumerate(product(theta, X)):
            y = + self.radius * np.sin(t)
            z = - self.radius * np.cos(t)
            nodes[i, :] = (x, y, z)
        nodes += -np.array([self.length/2, 0, 0])

        # Connectivities
        panels = np.zeros((ntheta//2, 4), dtype=int)

        for k, i in enumerate(range(0, ntheta//2)):
            panels[k, :] = (2*i, 2*i+2, 2*i+3, 2*i+1)
        half_ring = Mesh(nodes, panels, name=f"half_ring_of_{name}_mesh")

        if reflection_symmetry:
            if nx == 1:
                half_cylinder = half_ring
            else:
                half_cylinder = TranslationalSymmetricMesh(half_ring, translation=np.asarray([self.length / nx, 0.0, 0.0]),
                                                           nb_repetitions=nx-1, name=f"half_{name}_mesh")
                if not translation_symmetry:
                    half_cylinder = half_cylinder.merged()

            return ReflectionSymmetricMesh(half_cylinder, plane=xOz_Plane, name=f"{name}_mesh")

        else:
            strip = half_ring + half_ring.mirrored(plane=xOz_Plane)
            if nx == 1:
                return strip

            else:
                cylinder = TranslationalSymmetricMesh(strip, translation=np.asarray([self.length / nx, 0.0, 0.0]),
                                                      nb_repetitions=nx-1, name=f"half_{name}_mesh")
                if not translation_symmetry:
                    cylinder = cylinder.merged()

            return cylinder

    @property
    def volume(self):
        return self.length*np.pi*self.radius**2


class VerticalCylinder(FloatingBody):
    """Vertical cylinder.

    Parameters
    ----------
    length : float, optional
        length of the cylinder
    radius : float, optional
        radius of the cylinder
    center : 3-ple or array of shape (3,), optional
        position of the geometric center of the cylinder
    nx : int, optional
        number of circular slices
    ntheta : int, optional
        number of panels along a circular slice of the cylinder
    nr : int, optional
        number of panels along a radius on the extremities of the cylinder
    clever : bool, optional
        if True, uses the mesh symmetries
    name : str, optional
        a string naming the floating body
    """

    def __init__(self, length=10.0, radius=1.0, center=(0, 0, 0),
                 nx=10, ntheta=10, nr=2,
                 clever=True, name=None):
        self.length = length
        self.radius = radius
        self.geometric_center = np.asarray(center, dtype=float)

        if name is None:
            name = f"cylinder_{next(Mesh._ids)}"

        open_cylinder = AxialSymmetricMesh.from_profile(
            lambda z: radius,
            z_range=np.linspace(-length/2, length/2, nx+1),
            nphi=ntheta)

        if nr > 0:
            top_side = Disk(radius=radius, center=(0, 0, length/2),
                            axial_symmetry=True, normal=(0, 0, 1),
                            resolution=(nr, ntheta), name=f"top_side_of_{name}").mesh

            bottom_side = top_side.copy(name=f"bottom_side_of_{name}_mesh")
            bottom_side.mirror(xOy_Plane)

            mesh = AxialSymmetricMesh.join_meshes(open_cylinder, top_side, bottom_side)
        else:
            mesh = open_cylinder

        if not clever:
            mesh = mesh.merged()
            mesh.merge_duplicates()
            mesh.heal_triangles()

        mesh.translate(center)
        mesh.name = f"{name}_mesh"

        FloatingBody.__init__(self, mesh=mesh, name=name)

