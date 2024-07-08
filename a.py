from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.providers.aer import AerSimulator

def qft(circuit, n):
    """Apply QFT on the first n qubits in circuit."""
    for i in range(n):
        circuit.h(i)
        for j in range(i+1, n):
            circuit.cp(2 * 3.14159 / (2 ** (j-i+1)), i, j)
    circuit.barrier()

def quantum_hashing(binary_strings, selected_indices):
    # Initialize a list to hold the hashed values
    hashed_values = []
    
    for binary_string in binary_strings:
        n = len(binary_string)
        m = len(selected_indices)
        
        # Create a quantum circuit with n qubits
        qc = QuantumCircuit(n, m)
        
        # Initialize the quantum circuit with the binary string
        for i, bit in enumerate(reversed(binary_string)):
            if bit == '1':
                qc.x(i)
        
        # Apply QFT
        qft(qc, n)
        
        # Apply quantum hashing by measuring selected indices
        for idx, qubit in enumerate(selected_indices):
            qc.measure(qubit, idx)
        
        # Simulate the circuit
        simulator = AerSimulator()
        compiled_circuit = transpile(qc, simulator)
        qobj = assemble(compiled_circuit)
        result = simulator.run(qobj).result()
        counts = result.get_counts()
        measured_string = max(counts, key=counts.get)
        hashed_values.append(measured_string)
    
    return hashed_values

# Example usage
binary_strings = ["11010010", "01100101"]
selected_indices = [1, 3, 5]
hashed_values = quantum_hashing(binary_strings, selected_indices)
print("Hashed values:", hashed_values)
