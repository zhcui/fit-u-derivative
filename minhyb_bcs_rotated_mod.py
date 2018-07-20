#! /usr/bin/env python

import sys
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import newton,brentq 
from scipy.optimize import least_squares 
from scipy.optimize import leastsq
sys.path.append("../../") 
import utils
from meanfields import *
import scipy.linalg as la

def replicate_u_matrix( u_imp, Nbasis, Nimp, mtype):

    #Number of copies of impurity cluster in total lattice
    Ncopies = Nbasis / Nimp

    #Copy impurity u-matrix across entire lattice
    u_mat_replicate = np.zeros( [2*Nbasis,2*Nbasis],dtype=mtype)
    for cpy in range(0,Ncopies):
        u_mat_replicate[cpy*Nimp:Nimp+cpy*Nimp,cpy*Nimp:Nimp+cpy*Nimp] = u_imp[:Nimp,:Nimp]
        u_mat_replicate[cpy*Nimp+Nbasis:Nimp+cpy*Nimp+Nbasis,cpy*Nimp:Nimp+cpy*Nimp] = u_imp[Nimp:Nimp+Nimp,:Nimp]
        u_mat_replicate[cpy*Nimp:Nimp+cpy*Nimp,cpy*Nimp+Nbasis:Nimp+cpy*Nimp+Nbasis] = u_imp[:Nimp,Nimp:Nimp+Nimp]
        u_mat_replicate[cpy*Nimp+Nbasis:Nimp+cpy*Nimp+Nbasis,cpy*Nimp+Nbasis:Nimp+cpy*Nimp+Nbasis] = u_imp[Nimp:Nimp+Nimp,Nimp:Nimp+Nimp]
        
    return u_mat_replicate

def array2matrix( array, Nimp, mtype ):

    mat = np.zeros([Nimp*2, Nimp*2], mtype)
    
    p = 0            
    for i in range(2*Nimp):
        for j in range(i,2*Nimp):
            mat[i,j] = array[p]
            p += 1

    mat = mat + mat.conj().T 
    for i in range(2*Nimp):
        mat[i,i] *= 0.5
        
    return mat

def matrix2array( mat, Nimp):
        
    #Expects input in [2*Nimp,2*Nimp] format
    
    array = np.array([mat[i,j] for i in range(2*Nimp) for j in range(i,2*Nimp)],dtype=np.float64)

    #array = ([mat[i,j] for i in range(Nimp) for j in range(i,Nimp)])
    #array += ([mat[i,j+Nimp] for i in range(Nimp) for j in range(Nimp)])
    array = np.array(array,dtype=np.float64)

    return array


def minimizehybBCSR_mod(dmetfitIndex, h1emb, targetRDM, R, Nbasis, Nimp, u_mat0, mtype = np.float64, gtol = 1.0e-6, miter = 10):

    #Define parameters to optimize from previous DMET guess (vs guess from previous cluster in the loop ) of u-matrix associated with impurity sites in cluster
    #Note since u-matrix must be symmetric (if real) only optimize half the parameters
    #Note if constraining diagonal of u-matrix, first index of params will be the diagonal term, and the rest the upper triagonal of the u-matrix
    
    #TEST
    
    #Get the indices of interest 
    ll = np.triu_indices(dmetfitIndex)
    lln = len(ll[0])
 
    #h2el = hf.make_h2el(dmet.g2e_site,hf1rdm_site)    
    #fock = dmet.h1 + h2el  
    #h1body = dmet.h1 + dmet.fock2e + dmet.globalMu
    #h1emb = np.dot(R.conjugate().T,np.dot(h1body,R))
    #targetRDM = dmet.IRDM1

    #np.save('./targetRDM.npy', targetRDM) # ZHC TEMP NOTE
    #np.save('./R.npy', R) # ZHC TEMP NOTE
    #np.save('./h1emb.npy', h1emb) # ZHC TEMP NOTE
    
    #exit()

    

    def localdiag(N,h,u):
       ht = h+u
       e,c = la.eigh(ht) 
       cocc = c[:, :N]
       cvir = c[:, N:]
       return np.dot(cocc,cocc.conj().T), c, e, cocc, cvir
 
    def costf(x, A):

        #R = dmet.RotationMatrix
        u_mat_imp = array2matrix(x, Nimp, mtype)

        '''
        #Only adding to impurity?             
        uep = np.zeros_like(h1emb)
        uep[:R.shape[1]/2,:R.shape[1]/2] = u_mat_imp        
        '''
        #Also add to bath
        uep = np.dot(R.conjugate().T,np.dot(replicate_u_matrix(u_mat_imp, Nbasis, Nimp, mtype),R))

        num_occ = R.shape[1]/2

        hf1RDM_b, mo_coeff, mo_energy, mo_occ, mo_vir = localdiag(num_occ, h1emb, uep)

        


        #FIT
        impfit = (hf1RDM_b-targetRDM)[ll]

        print('Guess diff: %10.6e\n' %(np.linalg.norm(impfit)))


        # calc gradients
        de_ov = mo_energy[:num_occ][:,None] - mo_energy[num_occ:]
        
        B = np.einsum('km, ln, pn, qm, mn -> pqkl', mo_occ, mo_vir.conj(), mo_vir, mo_occ.conj(), de_ov)
        B += np.einsum('kn, lm, pn, qm, mn -> pqkl', mo_vir, mo_occ.conj(), mo_vir.conj(), mo_occ, de_ov)

        # TODO select independent indices
        B_ind = 




    
        return np.linalg.norm(impfit)**2 
 
#    def jacf(x):
#    
#        R = dmet.RotationMatrix
#        u_mat_imp = dmet.array2matrix(x)
#        
#        '''
#        #Add to impurity only    
#        uep = np.zeros_like(h1emb)
#        uep[:R.shape[1]/2,:R.shape[1]/2] = u_mat_imp        
#        '''
#        #Add to bath also
#        uep = np.dot(R.conjugate().T,np.dot(dmet.replicate_u_matrix(u_mat_imp),R))
#
#        hf1RDM, hforbs, hfevals = localdiag(R.shape[1]/2,h1emb,uep)
#        #Jacobian must be function by variables 
#        jac = np.zeros([len(x),R.shape[1],R.shape[1]], dtype = dmet.mtype)
#
#        #Only concerned with corners
#        for k in range(len(x)):
#            dV = np.zeros_like(x)
#            dV[k] = 1
#            dVm = dmet.array2matrix(dV)
#            vep = np.dot(R.conjugate().T,np.dot(dmet.replicate_u_matrix(dVm),R))
#    
#            #Get d\rho/dV[k]
#            drho = utils.analyticGradientO(hforbs,hfevals,vep,R.shape[1]/2)
#            jac[k] = drho
#
#        #Gradient function: careful with complex stuff
#        gradfn = np.zeros(len(x),dtype=np.float64)
#        diffrdm = (hf1RDM - dmet.IRDM1)[ll] 
#        diffrdmR = diffrdm.real
#        diffrdmI = diffrdm.imag
#        
#        for k in range(len(x)):
#            #gradfn[k] = np.sum(2.*np.multiply(jac[k][:dmet.fitIndex,:dmet.fitIndex],diffrdmR))
#            J = jac[k][ll]
#            gradfn[k] = 2.*np.sum(np.multiply(J.real,diffrdmR) + np.multiply(J.imag,diffrdmI))
#        
#        #print "Gradient: ",np.linalg.norm(gradfn)    
#        return gradfn
#
#    def costf_lsq(x):
#
#        R = dmet.RotationMatrix
#        u_mat_imp = dmet.array2matrix(x)
#
#        '''
#        #Only adding to impurity?             
#        uep = np.zeros_like(h1emb)
#        uep[:R.shape[1]/2,:R.shape[1]/2] = u_mat_imp        
#        '''
#        #Also add to bath
#        uep = np.dot(R.conjugate().T,np.dot(dmet.replicate_u_matrix(u_mat_imp),R))
#
#        hf1RDM_b,_,_ = localdiag(R.shape[1]/2,h1emb,uep)
#
#        #FIT
#        impfit = hf1RDM_b-targetRDM
#        residuals = utils.unpackComplex(impfit[ll])
#
#        if(dmet.debugPrintRDMDiff):
#            dmet.dlog.write('Guess diff: %10.6e\n' %(np.linalg.norm(impfit)))
#            dmet.dlog.flush()
# 
#        return residuals 
#
#    def jacf_lsq(x):
#    
#        R = dmet.RotationMatrix
#        u_mat_imp = dmet.array2matrix(x)
#        
#        '''
#        #Add to impurity only    
#        uep = np.zeros_like(h1emb)
#        uep[:R.shape[1]/2,:R.shape[1]/2] = u_mat_imp        
#        '''
#        #Add to bath also
#        uep = np.dot(R.conjugate().T,np.dot(dmet.replicate_u_matrix(u_mat_imp),R))
#
#        hf1RDM, hforbs, hfevals = localdiag(R.shape[1]/2,h1emb,uep)
#        #Jacobian must be function by variables 
#        jac = np.zeros([lln,len(x)], dtype = np.float64)
#
#        #Only concerned with corners
#        for k in range(len(x)):
#            dV = np.zeros_like(x)
#            dV[k] = 1
#            dVm = dmet.array2matrix(dV)
#            
#            '''
#            vep = np.zeros_like(h1emb)
#            vep[:R.shape[1]/2,:R.shape[1]/2] = dVm
#            '''
#
#            vep = np.dot(R.conjugate().T,np.dot(dmet.replicate_u_matrix(dVm),R))
#
#            #Get d\rho/dV[k]
#            drho = utils.analyticGradientO(hforbs,hfevals,vep,R.shape[1]/2)
#            erho = utils.unpackComplex(drho[ll].real)
#
#            jac[:,k] = erho
#
#        return jac 

    #First do BFGS
    u_mat_imp = utils.extractImp(Nimp, u_mat0)
    params = matrix2array(u_mat_imp, Nimp)    
    #Minimize difference between HF and correlated DMET 1RDMs
    min_result = minimize( costf, params, method = 'BFGS', jac = True , tol = gtol, options={'maxiter': miter*len(params), 'disp': True})
    #min_result = minimize( costf, params, method = 'BFGS', jac = jacf , tol = gtol, options={'maxiter': miter*len(params), 'disp': True})
    x = min_result.x
    
    print "BFGS Final Diff: ",min_result.fun**0.5,"Converged: ",min_result.status," Jacobian: ",np.linalg.norm(min_result.jac)      
    if(not min_result.success):
        print "WARNING: Minimization unsuccessful. Message: ",min_result.message
    
    jacr = np.linalg.norm(min_result.jac)
    '''
    if(jacr>gtol and jacr < gtol*1.5):
        #DO BFGS TO MAKE TIGHTER FIT
        #Minimize difference between HF and correlated DMET 1RDMs
        params = x 
        pbounds = ([-10.0 for y in params],[10.0 for y in params])
       
        min_result = least_squares(costf_lsq, params, ftol = gtol, jac = jacf_lsq, max_nfev = miter*len(params), bounds = pbounds)
        #min_result = least_squares(costf, params, ftol = gtol, max_nfev = miter*len(params), bounds = pbounds)
        x1 = min_result.x
        ier = min_result.status
        cost = min_result.cost
        print "LSQ Final Diff: ",cost,"Converged: ",ier," Gradient: ",np.linalg.norm(min_result.grad)
        print "Converge Msg: ",min_result.message    
        
        jacr2 = np.linalg.norm(min_result.grad)
        if(jacr2 < jacr):
            print "LSQ used."
            jacr = jacr2
            x = x1
    

    #Update new u-matrix from optimized parameters
    u_mat_imp_new = array2matrix(x, Nimp, mtype )
        
    lp = u_mat_imp_new.shape[0]/2
    #d_um = (u_mat_imp_new-u_mat_imp)[:lp,:lp].trace()/u_mat_imp.shape[0]                
    od_um = (u_mat_imp)[:lp,:lp].trace()/lp
    d_um = (u_mat_imp_new)[:lp,:lp].trace()/lp               
    print "Shift: ",d_um,d_um-od_um
    d_um -= od_um

    #Keep trace constrained
    if(dmet.itr>dmet.traceFixStart):
        print 'Fixing trace of u-matrix'
        u_mat_imp[:lp,:lp] = u_mat_imp_new[:lp,:lp] - np.eye(lp)*d_um
        u_mat_imp[lp:,lp:] = u_mat_imp_new[lp:,lp:] + np.eye(lp)*d_um
    else:
        u_mat_imp = u_mat_imp_new           

    dmet.u_mat_new = dmet.replicate_u_matrix( u_mat_imp )
    if dmet.dm_DIIS:
        R = dmet.RotationMatrix
        #u_mat_imp = u_mat_imp
        uep = np.dot(R.conjugate().T,np.dot(dmet.replicate_u_matrix(u_mat_imp),R))
        hf1RDM_b,_,_ = localdiag(R.shape[1]/2,h1emb,uep)
        dmet.dm_rot_new = hf1RDM_b
    '''

    return jacr
    #utils.displayMatrix(dmet.u_mat_new)
#####################################################################
 

if __name__ == '__main__':
    targetRDM = np.load('./targetRDM.npy') # ZHC TEMP NOTE
    R = np.load('./R.npy') # ZHC TEMP NOTE
    h1emb = np.load('./h1emb.npy') # ZHC TEMP NOTE
    u_mat0 = np.load('./u_mat.npy') 
    dmetfitIndex = R.shape[1] #includes imp+bath orbitals
    Nimp = 4
    Nbasis = R.shape[0]/2
    jacr = minimizehybBCSR_mod(dmetfitIndex, h1emb, targetRDM, R, Nbasis, Nimp, u_mat0, mtype = np.float64, gtol =
            1.0e-6, miter = 10) 
