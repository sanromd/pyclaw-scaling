#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import os
timeVec1 = PETSc.Vec().createWithArray([0])
timeVec2 = PETSc.Vec().createWithArray([0])
timeVec3 = PETSc.Vec().createWithArray([0])

def acoustics3D(finalt=1.0,use_petsc=True,outdir='./_output',solver_type='classic',disable_output=True,dm=[30,30,30],**kwargs):
    """
    Python script for scaling test based on 3D acoustics.
    """
    from clawpack import riemann

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if solver_type=='classic':
        solver=pyclaw.ClawSolver3D(riemann.vc_acoustics_3D)
        solver.limiters = pyclaw.limiters.tvd.MC
    elif solver_type=='sharpclaw':
        solver = pyclaw.SharpClawSolver3D(riemann.vc_acoustics_3D)
    else:
        raise Exception('Unrecognized solver_type.')

    size = PETSc.Comm.getSize(PETSc.COMM_WORLD)
    rank = PETSc.Comm.getRank(PETSc.COMM_WORLD)

    solver.bc_lower[0]=pyclaw.BC.periodic
    solver.bc_upper[0]=pyclaw.BC.periodic
    solver.bc_lower[1]=pyclaw.BC.periodic
    solver.bc_upper[1]=pyclaw.BC.periodic
    solver.bc_lower[2]=pyclaw.BC.periodic
    solver.bc_upper[2]=pyclaw.BC.periodic

    solver.aux_bc_lower[0]=pyclaw.BC.periodic
    solver.aux_bc_upper[0]=pyclaw.BC.periodic
    solver.aux_bc_lower[1]=pyclaw.BC.periodic
    solver.aux_bc_upper[1]=pyclaw.BC.periodic
    solver.aux_bc_lower[2]=pyclaw.BC.periodic
    solver.aux_bc_upper[2]=pyclaw.BC.periodic

    app = None
    # dm = None
    if 'test' in kwargs:
        test = kwargs['test']
        if test == 'homogeneous':
            app = 'test_homogeneous'
        elif test == 'heterogeneous':
            app = 'test_heterogeneous'
        else: raise Exception('Unrecognized test')

    if app == 'test_homogeneous':
        if solver_type=='classic':
            solver.dimensional_split=True
        else:
            solver.lim_type = 1

        solver.limiters = [4]
        # if 'griddm' in kwargs:
        #     dm=kwargs['griddm']
        mx=dm[0]; my=dm[1]; mz=dm[2]
        # else:
        #     mx=256; my=4; mz=4
        
        zr = 1.0  # Impedance in right half
        cr = 1.0  # Sound speed in right half

    if app == 'test_heterogeneous' or app == None:
        if solver_type=='classic':
            solver.dimensional_split=False
        
        solver.bc_lower[0]    =pyclaw.BC.wall
        solver.bc_lower[1]    =pyclaw.BC.wall
        solver.bc_lower[2]    =pyclaw.BC.wall
        solver.aux_bc_lower[0]=pyclaw.BC.wall
        solver.aux_bc_lower[1]=pyclaw.BC.wall
        solver.aux_bc_lower[2]=pyclaw.BC.wall
        # if 'griddm' in kwargs:
        #     dm=kwargs['griddm']
        mx=dm[0]; my=dm[1]; mz=dm[2]
        # else: 
        #     mx=30; my=30; mz=30
        
        zr = 2.0  # Impedance in right half
        cr = 2.0  # Sound speed in right half

    solver.limiters = pyclaw.limiters.tvd.MC
    solver.cfl_max = 0.5
    solver.cfl_desired = 0.45
    solver.dt_variable = True 
    
    # Initialize domain
    x = pyclaw.Dimension('x',-1.0,1.0,mx)
    y = pyclaw.Dimension('y',-1.0,1.0,my)
    z = pyclaw.Dimension('z',-1.0,1.0,mz)
    domain = pyclaw.Domain([x,y,z])

    num_eqn = 4
    num_aux = 2 # density, sound speed
    state = pyclaw.State(domain,num_eqn,num_aux)

    zl = 1.0  # Impedance in left half
    cl = 1.0  # Sound speed in left half

    grid = state.grid
    grid.compute_c_centers()
    X,Y,Z = grid._c_centers

    state.aux[0,:,:,:] = zl*(X<0.) + zr*(X>=0.) # Impedance
    state.aux[1,:,:,:] = cl*(X<0.) + cr*(X>=0.) # Sound speed

    x0 = -0.5; y0 = 0.; z0 = 0.
    if app == 'test_homogeneous':
        r = np.sqrt((X-x0)**2)
        width=0.2
        state.q[0,:,:,:] = (np.abs(r)<=width)*(1.+np.cos(np.pi*(r)/width))

    elif app == 'test_heterogeneous' or app == None:
        r = np.sqrt((X-x0)**2 + (Y-y0)**2 + (Z-z0)**2)
        width=0.1
        state.q[0,:,:,:] = (np.abs(r-0.3)<=width)*(1.+np.cos(np.pi*(r-0.3)/width))

    else: raise Exception('Unexpected application')
        
    state.q[1,:,:,:] = 0.
    state.q[2,:,:,:] = 0.
    state.q[3,:,:,:] = 0.

    dt=np.min(grid.delta)/2.0*solver.cfl_desired
    solver.dt = dt
    tic1 = MPI.Wtime()
    claw = pyclaw.Controller()
    claw.keep_copy = False
    if disable_output:
       claw.output_format = None
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.output_style = 3
    claw.nstepout = 1
    claw.outdir = outdir+'_'+str(size)

    # Solve
    claw.tfinal = finalt
    claw.num_output_times=1
    tic2 = MPI.Wtime()
    status = claw.run()
    toc = MPI.Wtime()
    timeVec1.array = toc - tic1
    timeVec2.array = toc - tic2
    t1 = MPI.Wtime()
    duration1 = timeVec1.max()[1]
    duration2 = timeVec2.max()[1]
    t2 = MPI.Wtime()
    log_save_file = os.path.join(outdir,"results_"+str(size)+'_'+str(mx)+".log")
    log = MPI.File.Open(MPI.COMM_WORLD, log_save_file, MPI.MODE_CREATE|MPI.MODE_WRONLY)

    
    if rank==0:
        results = np.empty([7])
        log.Write('clawrun + load took '+str(duration1)+' seconds, for process '+str(rank)+'\n')
        log.Write('clawrun took '+str(duration2)+' seconds, for process '+str(rank)+' in grid '+str(mx)+'\n')
        log.Write('number of steps: '+ str(claw.solver.status.get('numsteps'))+'\n')
        log.Write('time reduction time '+str(t2-t1)+'\n')
        log.Write('tfinal '+str(claw.tfinal)+' and dt '+str(solver.dt))
        print 'clawrun + load took '+str(duration1)+' seconds, for process '+str(rank)+'\n'
        print 'clawrun took '+str(duration2)+' seconds, for process '+str(rank)+' in grid '+str(mx)+'\n'
        print 'number of steps: '+ str(claw.solver.status.get('numsteps'))+'\n'
        print 'time reduction time '+str(t2-t1)+'\n'
        print 'tfinal '+str(claw.tfinal)+' and dt '+str(solver.dt)
        results[0] = size
        results[1] = mx
        results[2] = np.min(grid.delta)
        results[3] = claw.tfinal
        results[4] = solver.dt
        results[5] = duration1
        results[6] = duration2
        np.save(os.path.join(outdir,'results_'+str(size)),results)
        return results

if __name__=="__main__":
    import sys
    import os

    if sys.argv[2]=='sharpclaw' or sys.argv[2]=='classic':
        solver_type = str(sys.argv[2])
        if len(sys.argv)==4:
            out_folder = sys.argv[3]
        else:
            out_folder = '_output'
    else:
        solver_type = 'sharpclaw'
        out_folder = sys.argv[2]
        if rank==0:
            print 'assuming '+str(sys.argv[2])+' is a folder name'

    if not os.path.isdir(out_folder):
        if rank==0:
            os.mkdir(out_folder)

    size = PETSc.Comm.getSize(PETSc.COMM_WORLD) 
    tini = MPI.Wtime()
    acoustics3D(finalt=(0.2/np.sqrt(size))/40,dm=[sys.argv[1],sys.argv[1],sys.argv[1]],outdir=sys.argv[2])
    tfin = MPI.Wtime()
    dm = sys.argv[1]
    outdir_name=sys.argv[2]
    timeVec3.array = tfin - tini
    duration3 = timeVec3.max()[1]
    if MPI.COMM_WORLD.rank==0:
        results=np.load(os.path.join(outdir_name,'results_'+str(size)+'.npy'))
        results=np.append(results,duration3)
        print 'results total tiem '+str(duration3)
        np.save(os.path.join(outdir_name,'results_'+str(size)+'_'+str(dm)),results)