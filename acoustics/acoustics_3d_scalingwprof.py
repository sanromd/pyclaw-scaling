#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import os
timeVec1 = PETSc.Vec().createWithArray([0])
timeVec2 = PETSc.Vec().createWithArray([0])
timeVec3 = PETSc.Vec().createWithArray([0])

def acoustics3D(finalt=1.0,use_petsc=True,outdir='./_output',solver_type='sharpclaw',disable_output=True,dm=[30,30,30],**kwargs):
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

        mx=dm[0]; my=dm[1]; mz=dm[2]
        
        zr = 1.0  # Impedance in right half
        cr = 1.0  # Sound speed in right half

    if app == 'test_heterogeneous' or app == None:
        if solver_type=='classic':
            solver.dimensional_split=False
        
        solver.bc_lower[0] = pyclaw.BC.wall
        solver.bc_lower[1] = pyclaw.BC.wall
        solver.bc_lower[2] = pyclaw.BC.wall

        solver.aux_bc_lower[0] = pyclaw.BC.wall
        solver.aux_bc_lower[1] = pyclaw.BC.wall
        solver.aux_bc_lower[2] = pyclaw.BC.wall


        mx=dm[0]; my=dm[1]; mz=dm[2]

        
        zr = 2.0  # Impedance in right half
        cr = 2.0  # Sound speed in right half

    solver.cfl_max = 0.8
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

    tic1 = MPI.Wtime()
    claw = pyclaw.Controller()
    claw.keep_copy = False
    if disable_output:
       claw.output_format = None
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.output_style = 3
    claw.nstepout = 200
    claw.outdir = outdir+'_'+str(size)

    # Solve
    claw.tfinal = finalt
    claw.num_output_times = 1
    tic2 = MPI.Wtime()
    status = claw.run()
    toc = MPI.Wtime()
    timeVec1.array = toc - tic1
    timeVec2.array = toc - tic2
    t1 = MPI.Wtime()
    duration1 = timeVec1.max()[1]
    duration2 = timeVec2.max()[1]
    t2 = MPI.Wtime()
    log = MPI.File.Open(MPI.COMM_WORLD, os.path.join(outdir,"results_"+str(size)+'_'+str(ncells)+".log"), MPI.MODE_CREATE|MPI.MODE_WRONLY)

    
    if rank==0:
        results = np.empty([7])
        log.Write(solver_type+' \n')
        log.Write('clawrun + load took '+str(duration1)+' seconds, for process '+str(rank)+'\n')
        log.Write('clawrun took '+str(duration2)+' seconds, for process '+str(rank)+' in grid '+str(mx)+'\n')
        log.Write('number of steps: '+ str(claw.solver.status.get('numsteps'))+'\n')
        log.Write('time reduction time '+str(t2-t1)+'\n')
        log.Write('tfinal '+str(claw.tfinal)+' and dt '+str(solver.dt))
        print solver_type+' \n'        
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
        np.save(os.path.join(outdir,'results_'+str(size)+'_'+str(ncells)),results)
    return


if __name__=="__main__":
    tini = MPI.Wtime()
    import sys
    import os
    import cProfile
    
    size = PETSc.Comm.getSize(PETSc.COMM_WORLD)
    rank = PETSc.Comm.getRank(PETSc.COMM_WORLD)
    ncells = sys.argv[1]

    if sys.argv[2]=='sharpclaw' or sys.argv[2]=='classic':
        solver = str(sys.argv[2])
        if len(sys.argv)==4:
            out_folder = sys.argv[3]+'_'+solver
        else:
            out_folder = '_output_'+solver
    else:
        solver = 'sharpclaw'
        out_folder = sys.argv[2]+'_'+solver
        if rank==0:
            print 'assuming '+str(sys.argv[2])+' is a folder base name'

    if not os.path.isdir(out_folder):
        if rank==0:
            os.mkdir(out_folder)



    processList = list(np.arange(0,size,size/3))
    #out_folder = '/simdesk/clawpack/pyclaw/examples/acoustics_3d_variable/scaling_'+str(size)+'_'
    tb1call = MPI.Wtime()
    acoustics3D(finalt=(0.2/np.sqrt(size))/40,dm=[ncells,ncells,ncells],solver_type=solver,outdir=out_folder)
    PETSc.COMM_WORLD.barrier()

    tb2call = MPI.Wtime()    
    size = PETSc.Comm.getSize(PETSc.COMM_WORLD)
    rank = PETSc.Comm.getRank(PETSc.COMM_WORLD)
    acoustics3D(finalt=(0.2/np.sqrt(size))/40,dm=[ncells,ncells,ncells],solver_type=solver,outdir=out_folder)
    PETSc.COMM_WORLD.barrier()

    tb3call = MPI.Wtime()
    if rank in processList:
        funccall = "acoustics3D(finalt=(0.2/np.sqrt("+str(size)+"))/40,dm=["+str(ncells)+","+str(ncells)+","+str(ncells)+"],solver_type=solver,outdir=out_folder)"
        save_profile_name = os.path.join(out_folder,'statst_'+str(size)+'_'+str(ncells)+"_"+str(rank))
        cProfile.run(funccall,save_profile_name)
    else:
        print "process"+str(rank) +"not profiled"
        acoustics3D(finalt=(0.2/np.sqrt(size))/40,dm=[ncells,ncells,ncells],solver_type=solver,outdir=out_folder)
    PETSc.COMM_WORLD.barrier()
    tend = MPI.Wtime()
    timeVec3.array = tend - tini
    duration3 = timeVec3.max()[1]
    if MPI.COMM_WORLD.rank==0:
        results=np.load(os.path.join(out_folder,'results_'+str(size)+'_'+str(ncells)+'.npy'))
        results=np.append(results,duration3)
        results=np.append(results,tb1call - tini)
        results=np.append(results,tb3call - tb2call)
        results=np.append(results,tend - tb3call)
        results=np.append(results,2*(tb3call - tb2call) + (tend - tb3call))
        print 'results total time '+str(duration3)
        np.save(os.path.join(out_folder,'results_'+str(size)+'_'+str(ncells)),results)
        print "total time for proc",rank," is ",str(duration3), "up to before p1",tb1call-tini, "p1",tb2call-tb1call , "p2", tb3call-tb2call, "p3", tend-tb3call
        print "time to subtract from job time to give load time= part2*2+the rest of the code time", 2*(tb3call-tb2call)+ (tend- tb3call)