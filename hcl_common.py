import sys
if 'mfem.ser' in sys.modules:
    import mfem.ser as mfem
    from mfem.ser import HyperbolicElementFormIntegrator, HyperbolicFaceFormIntegrator
else:
    import mfem.par as mfem
    from mfem.par import HyperbolicElementFormIntegrator, HyperbolicFaceFormIntegrator
   
import numpy as np

class PyDGHyperbolicConservationLaws(mfem.TimeDependentOperator):
    def __init__(self, vfes:mfem.FiniteElementSpace, nonlinearForm:mfem.NonlinearForm, elementFormIntegrator:HyperbolicElementFormIntegrator, faceFormIntegrator:HyperbolicFaceFormIntegrator, num_equations_:int):
        super().__init__(vfes.GetNDofs()*num_equations_)
        self._vfes = vfes
        self._nonlinearForm = nonlinearForm
        self._elementFormIntegrator = elementFormIntegrator
        self._faceFormIntegrator = faceFormIntegrator
        self._num_equations = num_equations_
        
        self._max_char_speed = np.double(0.0)
        
        self._nonlinearForm.AddDomainIntegrator(self._elementFormIntegrator)
        
        self._nonlinearForm.AddInteriorFaceIntegrator(self._faceFormIntegrator)
        
        
        self._z = mfem.Vector(self.Height())
        
        self._computeElemMassInverse()
        
        # self._massInverseForm = mfem.BilinearForm(self._vfes)
        # self._massInverseForm.AddDomainIntegrator(mfem.InverseIntegrator(mfem.MassIntegrator()))
        # self._massInverseForm.Assemble()
    
    def _computeElemMassInverse(self):
        self._Me_inv = [mfem.DenseMatrix() for _ in range(self._vfes.GetNE())]
        mi = mfem.InverseIntegrator(mfem.MassIntegrator())
        for i in range(self._vfes.GetNE()):
            mi.AssembleElementMatrix(self._vfes.GetFE(i), self._vfes.GetElementTransformation(i), self._Me_inv[i])
        
    def getMaxCharSpeed(self):
        return self._max_char_speed
    
    def Mult(self, x:mfem.Vector, y:mfem.Vector):
        self._elementFormIntegrator.setMaxCharSpeed(0.0)
        self._faceFormIntegrator.setMaxCharSpeed(0.0)
        
        self._nonlinearForm.Mult(x, self._z)
        
        self._max_char_speed = max(self._elementFormIntegrator.getMaxCharSpeed(), self._faceFormIntegrator.getMaxCharSpeed())

        # 3. Multiply element-wise by the inverse mass matrices.
        zval = mfem.Vector()
        zmat = mfem.DenseMatrix()
        ymat = mfem.DenseMatrix()
        
        for i in range(self._vfes.GetNE()):
            # Return the vdofs ordered byNODES
            vdofs = mfem.intArray(self._vfes.GetElementVDofs(i))
            self._z.GetSubVector(vdofs, zval)
            dof = self._vfes.GetFE(i).GetDof()
            zmat.UseExternalData(zval.GetData(), dof, self._num_equations)
            ymat.SetSize(dof, self._num_equations)
            mfem.Mult(self._Me_inv[i], zmat, ymat)
            y.SetSubVector(vdofs, ymat.GetData())