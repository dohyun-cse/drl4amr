import mfem.ser as mfem
import numpy as np

print("Test for ordering of meshes. We expect all the below listed meshes are ordered as x-y-z")
print("This test ensures that we do not have to sort element centroids but just assume the ordering of square-meshes")
print()

print("Testing MakeCartesian1D")
mesh:mfem.Mesh = mfem.Mesh.MakeCartesian1D(10)
center = mfem.Vector(mesh.Dimension())
centers = mfem.DenseMatrix(mesh.Dimension(), mesh.GetNE())
for i in range(mesh.GetNE()):
    center.SetData(centers.GetColumn(i))
    mesh.GetElementCenter(i, center)
centers = centers.GetDataArray()
print(all([centers[i] < centers[i+1] for i in range(len(centers)-1)]))

print("Testing MakeCartesian2D - SFC False")
mesh:mfem.Mesh = mfem.Mesh.MakeCartesian2D(10, 20, mfem.Element.QUADRILATERAL, sfc_ordering=False)
center = mfem.Vector(mesh.Dimension())
centers = mfem.DenseMatrix(mesh.Dimension(), mesh.GetNE())
for i in range(mesh.GetNE()):
    center.SetData(centers.GetColumn(i))
    mesh.GetElementCenter(i, center)
arr = np.array([[1,2]])@centers.GetDataArray()
print(all([arr[i] < arr[i+1] for i in range(len(arr)-1)]))

print("Testing MakeCartesian3D - SFC False")
mesh:mfem.Mesh = mfem.Mesh.MakeCartesian3D(10, 20, 30, mfem.Element.HEXAHEDRON, sfc_ordering=False)
center = mfem.Vector(mesh.Dimension())
centers = mfem.DenseMatrix(mesh.Dimension(), mesh.GetNE())
for i in range(mesh.GetNE()):
    center.SetData(centers.GetColumn(i))
    mesh.GetElementCenter(i, center)
arr = np.array([[1,2,4]])@centers.GetDataArray()
print(all([arr[i] < arr[i+1] for i in range(len(arr)-1)]))

print("Testing Periodic-Square-4x4.mesh")
mesh:mfem.Mesh = mfem.Mesh("./mesh/periodic-square-4x4.mesh")
center = mfem.Vector(mesh.Dimension())
centers = mfem.DenseMatrix(mesh.Dimension(), mesh.GetNE())
for i in range(mesh.GetNE()):
    center.SetData(centers.GetColumn(i))
    mesh.GetElementCenter(i, center)
arr = np.array([[1,2]])@centers.GetDataArray()
print(all([arr[i] < arr[i+1] for i in range(len(arr)-1)]))