from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_bloch_vector
from qiskit_aer import Aer
from io import BytesIO

import base64
import tempfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


def array_to_bit_string(array):
    """Convert an array of bits into a bit string representation."""
    return ''.join(str(bit) for bit in array)


def bit_string_to_array(bit_string):
    """Convert a bit string representation into an array of bits."""
    return [int(bit) for bit in bit_string]


@st.cache_data
def generate_string(string_1, string_2, mode='matching'):
    """Generates a string based on the mode parameter."""
    if mode == 'matching':

        return ''.join(['Y' if char_1 == char_2 else '_' for char_1, char_2 in zip(string_1, string_2)])
    
    elif mode == 'unmatching':

        return ''.join(['X' if char_1 != char_2 else '_' for char_1, char_2 in zip(string_1, string_2)])
    
    elif mode == 'error':

        return ''.join(['X' if char_1 == 'Y' and char_2 == '_' else '_' 
                        for char_1, char_2 in zip(string_1, string_2)])
    
    elif mode == 'lucky':

        return ''.join(['Y' if char_1 == 'Y' and char_2 == 'Y' else '_' 
                        for char_1, char_2 in zip(string_1, string_2)])
    
    elif mode == 'mixed_matching':

        matching_string = []
        for char_1, char_2 in zip(string_1, string_2):
            if char_1 == ' ' or char_2 == ' ':  # No se considera, es un espacio en blanco
                matching_string.append('_')
            elif char_1 == char_2:  # Coinciden los caracteres
                matching_string.append('Y')
            else:  # No coinciden los caracteres
                matching_string.append('X')

        return ''.join(matching_string)
    
    elif mode == 'sifted_key':

        return ''.join([char_1 if char_2 == 'Y' else ' ' 
                        for char_1, char_2 in zip(string_1, string_2)])
    
    else:

        raise ValueError("Invalid mode. Choose from 'matching', 'unmatching', 'error', 'lucky', 'mixed_matching', or 'sifted_key'.")
    

def introduce_noise(bob_bits, error_rate):
    """Introduce noise by flipping some of Bob's bits with a given error rate."""
    noisy_bob_bits = []
    for bit in bob_bits:
        if np.random.rand() < error_rate:
            noisy_bob_bits.append(1 - bit)  # Flip the bit to introduce an error
        else:
            noisy_bob_bits.append(bit)
    return noisy_bob_bits


def generate_and_save_bloch_figures():
    # Definir los vectores de Bloch para cada estado
    bloch_vectors = {
        "|0⟩": [0, 0, 1],
        "|1⟩": [0, 0, -1],
        "|+⟩": [1, 0, 0],
        "|−⟩": [-1, 0, 0]
    }

    for state, vector in bloch_vectors.items():
        fig = plot_bloch_vector(vector, figsize=(2, 2), font_size=10)
        # Guardar la figura como imagen en el disco
        image_path = f"bloch_{state.replace('|', '').replace('⟩', '')}.png"
        fig.savefig(image_path, format='png', bbox_inches='tight')
        plt.close(fig)


def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


def visualize_statistics(bob_bits, alice_sifted_key_str, bob_sifted_key_str, revealed_bits_indexes, error_rate):
    """Visualize comprehensive statistics of the BB84 protocol and save relevant plots as images."""

    # Lista para almacenar las rutas de las imágenes
    image_paths = []

    # Convertir las claves filtradas de string a listas de bits
    sifted_keys = [(a, b) for a, b in zip(alice_sifted_key_str.replace(' ', ''), 
                                          bob_sifted_key_str.replace(' ', ''))]
    
    compared_keys = [(alice_sifted_key_str[i], bob_sifted_key_str[i]) for i in revealed_bits_indexes]

    # Gráfico de histograma de bits medidos por Bob
    plt.figure(figsize=(6, 4))
    hist, bins, patches = plt.hist(bob_bits, bins=2, edgecolor='black')
    plt.xticks([0.25, 0.75], ['0', '1'])  # Centra las marcas de los ticks para los bits de Bob
    plt.xlabel('Bob\'s Measured Bits')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Bob\'s Measured Bits')
    plt.ylim(0, max(hist) * 1.2)

    # Guardar la imagen temporalmente
    temp_file_histogram = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file_histogram.name)
    plt.close()
    image_paths.append(temp_file_histogram.name)

    # Gráfico de tasa de error
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(compared_keys)), [int(a != b) for a, b in compared_keys], 'ro-')
    plt.xlabel('Key Index')
    plt.ylabel('Error (1=Error, 0=Correct)')
    plt.title(f'Error Rate')
    plt.text(0.95, 0.95, f'Error Rate: {error_rate:.2%}',
             verticalalignment='top', horizontalalignment='right',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    
    temp_file_error_rate = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file_error_rate.name)
    plt.close()
    image_paths.append(temp_file_error_rate.name)

    # Gráfico de tasa de acuerdo de las claves filtradas
    plt.figure(figsize=(6, 4))
    agreement = [int(a == b) for a, b in sifted_keys]
    plt.bar(['Agree', 'Disagree'], [agreement.count(1), agreement.count(0)], color=['green', 'red'])
    plt.ylabel('Count')
    plt.title(f'Agreement Rate of Sifted Keys')
    for i, value in enumerate([agreement.count(1), agreement.count(0)]):
        plt.text(i, value, str(value), ha='center', va='bottom')
    
    temp_file_agreement = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file_agreement.name)
    plt.close()
    image_paths.append(temp_file_agreement.name)

    return image_paths


def create_bb84_circuits(n, alice_bits, alice_bases, bob_bases, eve_bases=None, eve_present=False):
    """Create a list of BB84 quantum circuits."""
    circuits = []
    for i in range(n):
        qc = QuantumCircuit(1, 1)

        # Alice's preparation
        if alice_bits[i] == 1:
            qc.x(0)
        if alice_bases[i] == 1:
            qc.h(0)
        qc.barrier()

        # Eve's eavesdropping
        if eve_present:
            if eve_bases[i] == 1:
                qc.h(0)
            qc.measure(0, 0)
            qc.barrier()

        # Bob's measurement
        if bob_bases[i] == 1:
            qc.h(0)
        qc.measure(0, 0)

        circuits.append(qc)

    return circuits


def quantum_transmission(n, alice_bits, alice_bases, bob_bases, eve_bases=None, eve_present=False, noisy_channel=False, error_rate=10, shots=1024):

    if eve_present:
        circuits = create_bb84_circuits(n, alice_bits, alice_bases, bob_bases, eve_bases, eve_present=True)
    else:
        circuits = create_bb84_circuits(n, alice_bits, alice_bases, bob_bases, eve_bases)

    simulator = Aer.get_backend('qasm_simulator')
    transpiled_circuits = transpile(circuits, simulator)
    results = simulator.run(transpiled_circuits, shots=shots).result()
    counts = [results.get_counts(circ) for circ in circuits]

    # Determine Bob's bit based on the majority vote
    bob_bits = np.array([int(max(count, key=count.get)) for count in counts])

    if noisy_channel:
        bob_bits = introduce_noise(bob_bits, error_rate/100)

    return bob_bits, counts
