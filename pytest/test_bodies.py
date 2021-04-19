#!/usr/bin/env python
# coding: utf-8

import pytest

import numpy as np

from capytaine.bodies import FloatingBody
from capytaine.meshes.meshes import Mesh
from capytaine.meshes.geometry import Axis, Plane
from capytaine.bodies.predefined.spheres import Sphere
from capytaine.bodies.predefined.cylinders import HorizontalCylinder


def test_dof():
    nodes = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]])
    faces = np.array([[0, 1, 2, 3]])
    body = FloatingBody(Mesh(nodes, faces), name="one_face")
    assert body.dofs == {}

    body.add_translation_dof(direction=(1.0, 0.0, 0.0), name="1")
    assert np.allclose(body.dofs["1"], np.array([1.0, 0.0, 0.0]))

    body.add_translation_dof(direction=(0.0, 1.0, 0.0), name="2")
    assert np.allclose(body.dofs["2"], np.array([0.0, 1.0, 0.0]))

    body.add_rotation_dof(Axis(vector=(0.0, 0.0, 1.0)), name="3")
    body.add_rotation_dof(Axis(point=(0.5, 0, 0), vector=(0.0, 0.0, 1.0)), name="4")


def test_dof_name_inference():
    body = HorizontalCylinder()
    body.add_translation_dof(direction=(1, 0, 0), name="Surge_1")
    for dofname in ['Surge', 'SURGE', 'surge']:
        body.add_translation_dof(name=dofname)
        assert np.allclose(body.dofs[dofname], body.dofs['Surge_1'])

    body.add_rotation_dof(name="Pitch")
    body.add_rotation_dof(name="yaw")

    body.dofs.clear()
    body.add_all_rigid_body_dofs()


def test_bodies():
    body = Sphere(name="sphere", clever=False)
    assert str(body) == "sphere"
    repr(body)
    assert np.allclose(body.geometric_center, (0, 0, 0))
    body.add_translation_dof(name="Surge")
    body.add_translation_dof(name="Heave")

    # Extract faces
    body.extract_faces(np.where(body.mesh.faces_centers[:, 2] < 0)[0])

    # Clipping
    body.keep_immersed_part(inplace=False)

    # Mirror of the dofs
    mirrored = body.mirrored(Plane(point=(1, 0, 0), normal=(1, 0, 0)))
    assert np.allclose(mirrored.geometric_center, np.array([2, 0, 0]))
    assert np.allclose(body.dofs['Surge'], -mirrored.dofs['Surge'])

    # Rotation of the dofs
    sideways = body.rotated(Axis(point=(0, 0, 0), vector=(0, 1, 0)), np.pi/2)
    assert np.allclose(sideways.dofs['Heave'][0], np.array([1, 0, 0]))

    upside_down = body.rotated(Axis(point=(0, 0, 0), vector=(0, 1, 0)), np.pi)
    assert np.allclose(body.dofs['Heave'], -upside_down.dofs['Heave'])

    # Copy of the body
    copy_of_body = body.copy(name="copy_of_sphere")
    copy_of_body.translate_x(10.0)
    copy_of_body.add_translation_dof(name="Heave")

    # Join bodies
    both = body.join_bodies(copy_of_body)
    assert set(both.dofs) == {'sphere__Surge', 'copy_of_sphere__Surge', 'sphere__Heave', 'copy_of_sphere__Heave'}


@pytest.mark.parametrize("z_center", [0, 2, -2])
@pytest.mark.parametrize("collection_of_meshes", [True, False])
def test_clipping_of_dofs(z_center, collection_of_meshes):
    """Check that clipping a body with a dof is the same as clipping the body ant then adding the dof."""
    full_sphere = Sphere(center=(0, 0, z_center), name="sphere", clever=collection_of_meshes, clip_free_surface=False)
    axis = Axis(point=(1, 0, 0), vector=(1, 0, 0))

    full_sphere.add_rotation_dof(axis, name="test_dof")
    clipped_sphere = full_sphere.keep_immersed_part(free_surface=0.0, sea_bottom=-np.infty, inplace=False)

    other_clipped_sphere = FloatingBody(mesh=clipped_sphere.mesh, name="other_sphere")
    other_clipped_sphere.add_rotation_dof(axis, name="test_dof")

    if clipped_sphere.mesh.nb_faces > 0:
        assert np.allclose(clipped_sphere.dofs['test_dof'], other_clipped_sphere.dofs['test_dof'])
    else:
        assert len(clipped_sphere.dofs['test_dof']) == 0


def test_mincing():
    body = HorizontalCylinder(length=10, radius=0.5, translation_symmetry=False, reflection_symmetry=False)
    body = body.minced((4, 1, 1))
    assert len(body.mesh) == 2
    assert np.all(body.mesh[0].faces_centers[:, 0] < 0)
    assert isinstance(body.mesh[0][0], Mesh)
    body = body.minced((1, 2, 2))
    assert isinstance(body.mesh[0][0][0][0], Mesh)

