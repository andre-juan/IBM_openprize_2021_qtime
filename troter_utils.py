import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})  # enlarge matplotlib fonts

# Import qubit states Zero (|0>) and One (|1>), and Pauli operators (X, Y, Z)
from qiskit.opflow import Zero, One, I, X, Y, Z

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Importing standard Qiskit modules
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, IBMQ, execute, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.monitor import job_monitor
from qiskit.circuit import Parameter

# Import state tomography modules
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.quantum_info import state_fidelity

import time

#################################################################
# ============================================================= #
#################################################################

def show_figure(fig):
    '''
    auxiliar function to display plot 
    even if it's not the last command of the cell
    from: https://github.com/Qiskit/qiskit-terra/issues/1682
    '''
    
    new_fig = plt.figure()
    new_mngr = new_fig.canvas.manager
    new_mngr.canvas.figure = fig
    fig.set_canvas(new_mngr.canvas)
    plt.show(fig)
    
#################################################################
# ============================================================= #
#################################################################

def show_decompose(qc, n):
    
    if n <= 0:
        
        show_figure(qc.draw("mpl"))
        print("#"*80)
        
    else:
        
        for _ in range(n):

            qc = qc.decompose()

            show_figure(qc.draw("mpl"))
            print("#"*80)
            
#################################################################
# ============================================================= #
#################################################################

from qiskit.compiler import transpile

def transpile_jakarta(qc, sim_noisy_jakarta, show_fig=False):

    for opt_level in range(4):

        transp_circ = transpile(qc, sim_noisy_jakarta, optimization_level=opt_level)

        if show_fig:
            
            show_figure(transp_circ.draw("mpl"))

        print(f'Optimization Level {opt_level}')
        print(f'Depth: {transp_circ.depth()}')
        print(f'Gate counts: {transp_circ.count_ops()}')
        print(f'Total number of gates: {sum(transp_circ.count_ops().values())}')

        print()
        print("#"*80)
        print()
        
        
#################################################################
# ============================================================= #
#################################################################

def XX(parameter):
    '''
    parameter: Parameter object
    '''

    # Build a subcircuit for XX(t) two-qubit gate
    XX_qr = QuantumRegister(2)
    XX_qc = QuantumCircuit(XX_qr, name='XX')

    XX_qc.ry(np.pi/2,[0,1])
    XX_qc.cnot(0,1)
    XX_qc.rz(2 * parameter, 1)
    XX_qc.cnot(0,1)
    XX_qc.ry(-np.pi/2,[0,1])

    # Convert custom quantum circuit into a gate
    XX = XX_qc.to_instruction()
    
    return XX

##############################################################
##############################################################

def YY(parameter):
    '''
    parameter: Parameter object
    '''
    
    # Build a subcircuit for YY(t) two-qubit gate
    YY_qr = QuantumRegister(2)
    YY_qc = QuantumCircuit(YY_qr, name='YY')

    YY_qc.rx(np.pi/2,[0,1])
    YY_qc.cnot(0,1)
    YY_qc.rz(2 * parameter, 1)
    YY_qc.cnot(0,1)
    YY_qc.rx(-np.pi/2,[0,1])

    # Convert custom quantum circuit into a gate
    YY = YY_qc.to_instruction()
    
    return YY

##############################################################
##############################################################

def ZZ(parameter):
    '''
    parameter: Parameter object
    '''

    # Build a subcircuit for ZZ(t) two-qubit gate
    ZZ_qr = QuantumRegister(2)
    ZZ_qc = QuantumCircuit(ZZ_qr, name='ZZ')

    ZZ_qc.cnot(0,1)
    ZZ_qc.rz(2 * parameter, 1)
    ZZ_qc.cnot(0,1)

    # Convert custom quantum circuit into a gate
    ZZ = ZZ_qc.to_instruction()
    
    return ZZ

#################################################################
# ============================================================= #
#################################################################

def trotter_first_order(parameter):
    '''
    parameter: Parameter object
    '''
    
    num_qubits = 3

    Trot_qr = QuantumRegister(num_qubits)
    Trot_qc = QuantumCircuit(Trot_qr, name='Trot')

    for i in range(0, num_qubits - 1):
        
        Trot_qc.append(ZZ(parameter), [Trot_qr[i], Trot_qr[i+1]])
        Trot_qc.append(YY(parameter), [Trot_qr[i], Trot_qr[i+1]])
        Trot_qc.append(XX(parameter), [Trot_qr[i], Trot_qr[i+1]])

    Trot_gate = Trot_qc.to_instruction()
    
    return Trot_gate

#################################################################
# ============================================================= #
#################################################################

def trotter_second_order(parameter):
    '''
    parameter: Parameter object
    '''
    
    num_qubits = 3

    Trot_qr = QuantumRegister(num_qubits)
    Trot_qc = QuantumCircuit(Trot_qr, name='Trot')

    Trot_qc.append(ZZ(parameter/2), [Trot_qr[0], Trot_qr[1]])
    Trot_qc.append(YY(parameter/2), [Trot_qr[0], Trot_qr[1]])
    Trot_qc.append(XX(parameter/2), [Trot_qr[0], Trot_qr[1]])
    
    Trot_qc.append(ZZ(parameter), [Trot_qr[1], Trot_qr[2]])
    Trot_qc.append(YY(parameter), [Trot_qr[1], Trot_qr[2]])
    Trot_qc.append(XX(parameter), [Trot_qr[1], Trot_qr[2]])
    
    Trot_qc.append(ZZ(parameter/2), [Trot_qr[0], Trot_qr[1]])
    Trot_qc.append(YY(parameter/2), [Trot_qr[0], Trot_qr[1]])
    Trot_qc.append(XX(parameter/2), [Trot_qr[0], Trot_qr[1]])

    Trot_gate = Trot_qc.to_instruction()
    
    return Trot_gate

#################################################################
# ============================================================= #
#################################################################

def trotter_step(order, parameter):
    '''
    parameter: Parameter object
    '''
    
    if order == 1:
        
        Trot_gate = trotter_first_order(parameter)
        
    elif order == 2:
        
        Trot_gate = trotter_second_order(parameter)
        
    else:
        
        raise ValueError("Only 1st or 2nd orders allowed!")
        
    return Trot_gate

#################################################################
# ============================================================= #
#################################################################

def view_single_trotter_step(order, parameter):
    '''
    parameter: Parameter object
    '''
    
    num_qubits = 3
    qc = QuantumCircuit(num_qubits)

    qc.append(trotter_step(order, parameter), range(num_qubits))

    print("Single trotterization step:")
    show_decompose(qc, 1)
     

#################################################################
# ============================================================= #
#################################################################

def full_trotter_circ(order, trotter_steps=4, target_time=np.pi,
                      uniform_times=True, steps_times=None):
    '''
    args:
    - order: 1 or 2 for first or second order;
    - trotter_steps: number of steps, must be >=4;
    - target_time: final evolution must be t=pi, but added asa parameter, so we can simulate other times;
    - uniform: boolean indicating wheter or not uniform times will be used;
    - steps_times: list with times for each step, in order. length must be trotter_steps!
    '''

    # Initialize quantum circuit for 3 qubits
    qr = QuantumRegister(7)
    qc = QuantumCircuit(qr)

    # Prepare initial state (remember we are only evolving 3 of the 7 qubits on
    # jakarta qubits (q_5, q_3, q_1) corresponding to the state |110>)
    # DO NOT MODIFY (|q_5,q_3,q_1> = |110>)
    qc.x([3,5])  
    
    # in this case, all times parameter are the same, because all times will be
    # the same, equal to target_time/trotter_steps, as indicated below in bind_parameters
    if uniform_times:

        Trot_gate = trotter_step(order, Parameter('t'))

        for _ in range(trotter_steps):

            qc.append(Trot_gate, [qr[1], qr[3], qr[5]])
            
        # evaluate simulation at target_time (t=pi) meaning each trotter step evolves pi/trotter_steps in time
        qc = qc.bind_parameters({qc.parameters[0]: target_time/trotter_steps})
    
    # now, in this case, we'll have different times for each step
    # but such that they sum to target_time, of course.
    # these times are in the parameter "steps_times".
    # and, because they're different, we'll have different parameters as well!
    else:
        
        # check
        if len(steps_times) != trotter_steps:
            raise ValueError(f"Incorrect quantity of times {len(steps_times)}! Must be equal to number of steps {trotter_steps}")
                             
        for i in range(trotter_steps):
            
            Trot_gate = trotter_step(order, Parameter(f't{i}'))
                                     
            qc.append(Trot_gate, [qr[1], qr[3], qr[5]])
                                     
        params_dict = {param: time for param, time in zip(qc.parameters, steps_times)}
                                     
        qc = qc.bind_parameters(params_dict)
         

    return qr, qc

#################################################################
# ============================================================= #
#################################################################

def state_tomagraphy_circs(order, trotter_steps=4, target_time=np.pi,
                           uniform_times=True, steps_times=None):
    '''
    trotter_steps: number of steps, must be >=4
    order: 1 or 2 for first or second order
    '''
    
    qr, qc = full_trotter_circ(order, trotter_steps, target_time,
                               uniform_times, steps_times)

    # Generate state tomography circuits to evaluate fidelity of simulation
    st_qcs = state_tomography_circuits(qc, [qr[1], qr[3], qr[5]])

    return st_qcs

#################################################################
# ============================================================= #
#################################################################

def execute_st_simulator(st_qcs, backend):
    '''
    backend: preferably sim_noisy_jakarta or sim_no_noise
    '''
    
    shots = 8192
    reps = 8

    jobs = []
    print()
    for i in range(reps):
        # execute
        job = execute(st_qcs, backend, shots=shots)
        print(f'{i+1}/{reps} - Job ID', job.job_id())
        jobs.append(job)
        
    return jobs


#################################################################
# ============================================================= #
#################################################################

# Compute the state tomography based on the st_qcs quantum circuits and the results from those ciricuits
def state_tomo(result, st_qcs):
    
    # The expected final state; necessary to determine state tomography fidelity
    target_state = (One^One^Zero).to_matrix()  # DO NOT MODIFY (|q_5,q_3,q_1> = |110>)
    
    # Fit state tomography results
    tomo_fitter = StateTomographyFitter(result, st_qcs)
    
    rho_fit = tomo_fitter.fit(method='lstsq')
    
    # Compute fidelity
    fid = state_fidelity(rho_fit, target_state)
    
    return fid

#################################################################
# ============================================================= #
#################################################################

def final_fidelities(jobs, st_qcs, order, trotter_steps):
    
    # Compute tomography fidelities for each repetition
    fids = []
    
    for job in jobs:
        
        fid = state_tomo(job.result(), st_qcs)
        
        fids.append(fid)

    print()
    print("#"*80)
    print()
    print(f"Final results - order: {order} - strotter steps: {trotter_steps}\n")
    print('State tomography fidelity = {:.4f} \u00B1 {:.4f}'.format(np.mean(fids), np.std(fids)))
    
    return fids

#################################################################
# ============================================================= #
#################################################################

def simulate_full_circ(qc, backend):
    '''
    returns p(psi = 110) at the end of the evolution
    '''
    
    counts = execute(qc, backend, shots=1e5, seed_simulator=42).result().get_counts()

    return counts["110"]/sum(counts.values())

#################################################################
# ============================================================= #
#################################################################

def simulate_H_all_t(order, trotter_steps, backend,
                     uniform_times=True, steps_times=None):
    
    print()
    print("#"*80)
    print()
    
    print("Starting simulation for times from 0 to pi!")
    start = time.time()

    ts = np.linspace(0, np.pi, 100)
    probs = []
        
    for target_time in ts:
        
        if uniform_times:
            # keep as is, in this case, None
            steps_times_current = None
        else:          
            # re-normalize times, so that they sum to target_time
            steps_times_current = np.array(steps_times)*(target_time/sum(steps_times))

        st_qcs = state_tomagraphy_circs(order, trotter_steps, target_time,
                                        uniform_times, steps_times_current)

        # last one in state tomography is always the one in which
        # only the assigned qubits are measured
        prob_110 = simulate_full_circ(st_qcs[-1], backend)

        probs.append(prob_110)

    print("Simulation ended!")
    stop = time.time()
    duration = time.strftime("%H:%M:%S", time.gmtime(stop-start))

    print(f"Total time of simulation: {duration}\n")
    
    # fidelity (prob) at t=pi
    fidelity_pi = np.array(probs)[np.where(ts == np.pi)].squeeze()
    
    return ts, probs, fidelity_pi

#################################################################
# ============================================================= #
#################################################################

def plot_simulation_H_all_t(ts, probs, fidelity_pi, plot_theoretical=True):
      
    plt.plot(ts, probs, label="simulated")
    
    plt.xlabel('time')
    plt.ylabel(r'Prob. of state $|110\rangle$')
    plt.title(r'Evolution of $|110\rangle$ under $H_{Heis3}(t)$')
    plt.grid()

    plt.axhline(y=fidelity_pi, color="red", ls=":", label=f"F($\pi)={fidelity_pi}$")
    
    if plot_theoretical:
        
        # computed with opflow
        probs_theoretical = [1.0, 0.9959801051027279, 0.9840172692892207, 0.9643990582381565, 0.937594911988758, 0.9042417387878469,
                             0.865124429911968, 0.821151950110675, 0.7733298053832053, 0.7227298088581207, 0.670458152474241, 
                             0.6176228439841472, 0.5653015837895042, 0.5145111338492695, 0.46617917227787303, 0.42111953445810696, 
                             0.3800116179277513, 0.3433845784384221, 0.31160677383035945, 0.28488072684522886, 0.26324368434414297,
                             0.24657365551117913, 0.23460062242465882, 0.22692243956194447, 0.22302478059314604, 0.22230435674939336, 
                             0.22409452576938554, 0.22769233752892054, 0.23238602435861347, 0.23748194190916663, 0.2423300000849707, 
                             0.24634669159746717, 0.2490349254154318, 0.2500000000000001, 0.24896120190274262, 0.24575868345200233, 
                             0.24035545262059993, 0.2328344921370996, 0.22339120671060558, 0.21232157022640713, 0.20000650262308176, 
                             0.1868931431419818, 0.1734737977909714, 0.16026342019134074, 0.1477765335911936, 0.13650451605088834, 
                             0.126894150225658, 0.11932828465859018, 0.11410936717336749, 0.11144649610946522, 0.11144649610946515, 
                             0.1141093671733676, 0.11932828465859018, 0.12689415022565823, 0.13650451605088837, 0.14777653359119364, 
                             0.16026342019134082, 0.1734737977909711, 0.18689314314198185, 0.20000650262308153, 0.2123215702264073, 
                             0.22339120671060558, 0.23283449213709922, 0.24035545262059999, 0.24575868345200222, 0.2489612019027428,
                             0.24999999999999994, 0.24903492541543146, 0.24634669159746728, 0.2423300000849708, 0.23748194190916688, 
                             0.23238602435861336, 0.22769233752892049, 0.22409452576938554, 0.22230435674939347, 0.2230247805931461, 
                             0.2269224395619447, 0.2346006224246586, 0.24657365551117874, 0.26324368434414297, 0.2848807268452291, 
                             0.3116067738303597, 0.3433845784384221, 0.380011617927751, 0.42111953445810757, 0.46617917227787303, 
                             0.5145111338492697, 0.5653015837895042, 0.6176228439841478, 0.6704581524742411, 0.7227298088581207, 
                             0.7733298053832053, 0.8211519501106751, 0.8651244299119683, 0.9042417387878462, 0.9375949119887566, 
                             0.9643990582381559, 0.9840172692892216, 0.9959801051027278, 1.0000000000000004]
        
        plt.plot(ts, probs_theoretical, label="theoretical")

    plt.legend(prop={'size': 12}, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.show()
    
#################################################################
# ============================================================= #
#################################################################