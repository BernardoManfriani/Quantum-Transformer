import cudaq

@cudaq.kernel
def unitary(qubits: cudaq.qview, theta: list[float], subsystem_size: int, func: int, layers: int):
    start, end = 0, 0

    if func == -1:
        start, end = 0, qubits.size()
    elif func == 0:
        start, end = 0, subsystem_size
    elif func == 1:
        start, end = subsystem_size, 2 * subsystem_size
    elif func == 2:
        start, end = 2 * subsystem_size, 3 * subsystem_size

    for layer in range(layers):
        for i in range(start, end):
            idx_in_theta = i - start + layer * (end - start)
            angle = theta[idx_in_theta]
            ry(angle, qubits[i])  

        for i in range(start, end):
            if i < end - 1:
                x.ctrl(qubits[i], qubits[i + 1])   
            else:
                x.ctrl(qubits[i], qubits[start])  

@cudaq.kernel
def controlled_adjoint_unitary(
    control: cudaq.qubit,
    qubits: cudaq.qview,
    theta: list[float],
    subsystem_size: int,
    func: int,
    layers: int,
):
    start, end = 0, 0

    if func == -1:
        start, end = 0, qubits.size()
    elif func == 0:
        start, end = 0, subsystem_size
    elif func == 1:
        start, end = subsystem_size, 2 * subsystem_size
    elif func == 2:
        start, end = 2 * subsystem_size, 3 * subsystem_size

    for layer in range(layers - 1, -1, -1):
        if (end - start) > 1:
            x.ctrl(control, qubits[end - 1], qubits[start])  
            for i in range(end - 2, start - 1, -1):
                x.ctrl(control, qubits[i], qubits[i + 1])   

        for i in range(end - 1, start - 1, -1):
            angle_index = i - start + layer * (end - start)
            angle = -theta[angle_index]
            ry.ctrl(angle, control, qubits[i])  

@cudaq.kernel
def controlled_unitary(
    control: cudaq.qubit,
    qubits: cudaq.qview,
    theta: list[float],
    subsystem_size: int,
    func: int,
    layers: int,
):
    start, end = 0, 0

    if func == -1:
        start, end = 0, qubits.size()
    elif func == 0:
        start, end = 0, subsystem_size
    elif func == 1:
        start, end = subsystem_size, 2 * subsystem_size
    elif func == 2:
        start, end = 2 * subsystem_size, 3 * subsystem_size

    for layer in range(layers):
        for i in range(start, end):
            angle_index = i - start + layer * (end - start)
            angle = theta[angle_index]
            ry.ctrl(
                theta[angle_index],  # angle
                control,             # control qubit
                qubits[i]            # target qubit
            )  

        for i in range(start, end):
            if i < end - 1:
                x.ctrl(control, qubits[i], qubits[i + 1])  
            else:
                x.ctrl(control, qubits[i], qubits[start]) 


@cudaq.kernel
def build_sequence_only_circuit(
    token_i: list[float],
    position_i: list[float],
    query: list[float],
    token_j: list[float],
    position_j: list[float],
    key: list[float],
    ansatz_layers: list[int],
    num_working_qubits: list[int],
):
    layers = ansatz_layers[0]
    ancilla = cudaq.qubit()
    register = cudaq.qvector(num_working_qubits[0])
    subsystem_size = num_working_qubits[0] // 2

    h(ancilla)  # type: ignore  # noqa: F821
    unitary(register, token_i, subsystem_size, 0, layers)
    unitary(register, position_i, subsystem_size, 1, layers)
    unitary(register, query, subsystem_size, -1, layers)
    controlled_adjoint_unitary(ancilla, register, query, subsystem_size, -1, layers)
    controlled_adjoint_unitary(ancilla, register, position_i, subsystem_size, 1, layers)
    controlled_adjoint_unitary(ancilla, register, token_i, subsystem_size, 0, layers)
    controlled_unitary(ancilla, register, token_j, subsystem_size, 0, layers)
    controlled_unitary(ancilla, register, position_j, subsystem_size, 1, layers)
    controlled_unitary(ancilla, register, key, subsystem_size, -1, layers)
    h(ancilla)  # type: ignore  # noqa: F821
