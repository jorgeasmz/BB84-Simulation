from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_bloch_vector
from qiskit_aer import Aer

import time
import tempfile
import streamlit as st
import numpy as np
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

        return ''.join(['✓' if char_1 == char_2 else '_' for char_1, char_2 in zip(string_1, string_2)])
    
    elif mode == 'unmatching':

        return ''.join(['✗' if char_1 != char_2 else '_' for char_1, char_2 in zip(string_1, string_2)])
    
    elif mode == 'error':

        return ''.join(['✗' if char_1 == '✓' and char_2 == '_' else '_' 
                        for char_1, char_2 in zip(string_1, string_2)])
    
    elif mode == 'lucky':

        return ''.join(['✓' if char_1 == '✓' and char_2 == '✓' else '_' 
                        for char_1, char_2 in zip(string_1, string_2)])
    
    elif mode == 'mixed_matching':

        matching_string = []
        for char_1, char_2 in zip(string_1, string_2):
            if char_1 == ' ' or char_2 == ' ':  # No se considera, es un espacio en blanco
                matching_string.append('_')
            elif char_1 == char_2:  # Coinciden los caracteres
                matching_string.append('✓')
            else:  # No coinciden los caracteres
                matching_string.append('✗')

        return ''.join(matching_string)
    
    elif mode == 'sifted_key':

        return ''.join([char_1 if char_2 == '✓' else ' ' 
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


@st.cache_resource
def generate_bloch_figures():
    # Definir los vectores de Bloch para cada estado
    bloch_vectors = {
        "|0⟩": [0, 0, 1],
        "|1⟩": [0, 0, -1],
        "|+⟩": [1, 0, 0],
        "|−⟩": [-1, 0, 0]
    }

    # Crear un diccionario para almacenar las figuras
    bloch_figures = {}

    for state, vector in bloch_vectors.items():
        fig = plot_bloch_vector(vector, figsize=(2, 2), font_size=10)
        bloch_figures[state] = fig
    
    return bloch_figures


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


def quantum_transmission(n, alice_bits, alice_bases, bob_bases, eve_bases=None, eve_present=False, noisy_channel=False, shots=1024):

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
        bob_bits = introduce_noise(bob_bits, 0.1)

    return bob_bits, counts


#

# Título de la aplicación
st.title('BB84 Protocol')

# Emojis y texto al inicio del sidebar
st.sidebar.markdown("<h2>BB84 Protocol</h2>", unsafe_allow_html=True)
st.sidebar.write("Explore a guided simulation of the BB84 protocol.")

# Menú en la barra lateral con emojis
menu = st.sidebar.radio("Select an option", 
                        ["🔬 Simulation", 
                         "📚 About the protocol"])

# Descripción según la opción seleccionada
if menu == "🔬 Simulation":

    st.sidebar.write("Run the BB84 protocol simulation and visualize the results.")

elif menu == "📚 About the protocol":

    st.sidebar.write("Find detailed information about the BB84 protocol.")


# Información adicional en letra pequeña
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <small>
    <a href="https://github.com/jorgeasmz/BB84-Simulation">BB84 Protocol</a> |
    Aug 2024 |
    Jorge Arias
    </small>
    """,
    unsafe_allow_html=True
)

# Descripción según la opción seleccionada
if menu == "🔬 Simulation":

    # Sección 0: Initial parameters
    st.write('### Initial parameters')

    n = st.slider('Number of qubits', min_value=1, max_value=100, value=50,  help="Select the amount of qubits for the simulation.")

    # Uso de radio button para seleccionar el escenario
    scenario = st.radio(
        "Select scenario:", ('Standard', 'Noisy channel', 'Eavesdropping'),
        index=0,
        help="Choose the scenario you want to simulate."
    )

    # Configuración basada en la selección
    noisy_channel = False
    eve_present = False

    if scenario == 'Noisy channel':
        noisy_channel = True
    elif scenario == 'Eavesdropping':
        eve_present = True


    # Función para inicializar estados de la simulación
    def init_simulation():
        if 'section' not in st.session_state:
            st.session_state.section = 0
        if 'alice_animation_running' not in st.session_state:
            st.session_state.alice_animation_running = False
        if 'bob_animation_running' not in st.session_state:
            st.session_state.bob_animation_running = False
        if 'alice_bits_str' not in st.session_state:
            st.session_state.alice_bits_str = ''
        if 'alice_bases_str' not in st.session_state:
            st.session_state.alice_bases_str = ''
        if 'bob_bases_str' not in st.session_state:
            st.session_state.bob_bases_str = ''


    # Inicializar estados de la simulación
    init_simulation()


    if st.button('Execute simulation'):

        st.session_state.section = 1
        st.session_state.alice_animation_running = True


    # Sección 1: Quantum transmission
    if st.session_state.section >= 1:

        st.write('### Quantum transmission')

        placeholder_alice_bits = st.empty()
        placeholder_alice_bases = st.empty()

        # Mostrar los bits actuales
        placeholder_alice_bits.text(
            f"Alice's Bits  : {st.session_state.alice_bits_str}")
        placeholder_alice_bases.text(
            f"Alice's Bases : {st.session_state.alice_bases_str}")

        # Botón para detener la animación
        if st.button('Generate bit strings'):

            st.session_state.alice_animation_running = False

            st.session_state.alice_bits_str = ''.join(str(np.random.randint(2)) for _ in range(n))
            st.session_state.alice_bases_str = ''.join(str(np.random.randint(2)) for _ in range(n))

            # Mostrar los bits actuales después de detener la animación
            placeholder_alice_bits.text(
                f"Alice's Bits  : {st.session_state.alice_bits_str}")
            placeholder_alice_bases.text(
                f"Alice's Bases : {st.session_state.alice_bases_str}")

            st.session_state.section = 2


        while st.session_state.alice_animation_running:

            # Generar nuevos bits de manera aleatoria para la animación
            st.session_state.alice_bits_str = ''.join(str(np.random.randint(2)) for _ in range(n))
            st.session_state.alice_bases_str = ''.join(str(np.random.randint(2)) for _ in range(n))

            # Mostrar en el placeholder
            placeholder_alice_bits.text(
                f"Alice's Bits  : {st.session_state.alice_bits_str}")
            placeholder_alice_bases.text(
                f"Alice's Bases : {st.session_state.alice_bases_str}")

            time.sleep(0.1)  # Ajusta la velocidad de la animación


    # Sección 2: Show qubits to prepare
    if st.session_state.section >= 2:

        if st.button('Show qubits to prepare'):

            st.session_state.section = 3
            st.session_state.bob_animation_running = True


    # Sección 3: Bit encoding
    if st.session_state.section >= 3:

        st.write("#### Prepared qubits")
        qubits_to_prepare = [
            "$$|0⟩$$" if base == "0" and bit == "0" else
            "$$|1⟩$$" if base == "0" and bit == "1" else
            "$$|+⟩$$" if base == "1" and bit == "0" else
            "$$|−⟩$$"
            for base, bit in zip(st.session_state.alice_bases_str, st.session_state.alice_bits_str)
        ]

        st.write(" ".join(qubits_to_prepare))

        placeholder_bob_bases = st.empty()

        # Mostrar los bits actuales
        placeholder_bob_bases.text(f"Bob's Bases: {st.session_state.bob_bases_str}")

        # Botón para detener la animación
        if st.button('Generate bit string'):

            st.session_state.bob_animation_running = False

            st.session_state.bob_bases_str = ''.join(str(np.random.randint(2)) for _ in range(n))

            # Mostrar los bits actuales después de detener la animación
            placeholder_bob_bases.text(f"Bob's Bases: {st.session_state.bob_bases_str}")

            st.session_state.section = 4


        while st.session_state.bob_animation_running:

            # Generar bases de manera aleatoria para la animación de Bob
            st.session_state.bob_bases_str = ''.join(str(np.random.randint(2)) for _ in range(n))

            # Mostrar en el placeholder
            placeholder_bob_bases.text(f"Bob's Bases: {st.session_state.bob_bases_str}")

            time.sleep(0.1)  # Ajusta la velocidad de la animación


    # Sección 4: Show measured qubits
    if st.session_state.section >= 4:

        alice_bits = bit_string_to_array(st.session_state.alice_bits_str)
        alice_bases = bit_string_to_array(st.session_state.alice_bases_str)
        bob_bases = bit_string_to_array(st.session_state.bob_bases_str)
        eve_bases = None

        if eve_present and 'eve_bases_str' not in st.session_state:

            st.session_state.eve_bases_str = ''.join(str(np.random.randint(2)) for _ in range(n))
            eve_bases = bit_string_to_array(st.session_state.eve_bases_str)

        if 'bob_bits_str' not in st.session_state:
            
            bob_bits, _ = quantum_transmission(n, alice_bits, alice_bases, bob_bases, eve_bases=eve_bases, eve_present=eve_present, noisy_channel=noisy_channel)

            st.session_state.bob_bits_str = array_to_bit_string(bob_bits)

        if st.button('Show measured qubits'):

            st.session_state.section = 5


    # Sección 5: Bit decoding
    if st.session_state.section >= 5:

        st.write("#### Obtained bits")

        placeholder_bob_bases = st.empty()

        # Mostrar los bits actuales
        placeholder_bob_bases.text(f"Bob's Bits: {st.session_state.bob_bits_str}")
        
        if st.button("Post-processing"):

            st.session_state.section = 6


    # Sección 6: Classical post-processing
    if st.session_state.section >= 6:

        st.write('### Classical post-processing')

        # Alice and Bob bases information
        if 'alice_and_bob_matching_bases_str' not in st.session_state: 
            
            matching_bases = generate_string(
                st.session_state.alice_bases_str, 
                st.session_state.bob_bases_str,
                mode='matching')
        
            st.session_state.alice_and_bob_matching_bases_str = matching_bases


        st.markdown(f"""
        <div style="overflow-x: auto; white-space: pre; font-family: monospace; font-size: 13px; margin-bottom: 10px;">Alice's Bases      : {st.session_state.alice_bases_str}
        Bob's Bases        : {st.session_state.bob_bases_str}
        A-B Matching Bases : {st.session_state.alice_and_bob_matching_bases_str}
        </div>
        """, unsafe_allow_html=True)

        guessed_bases_bob = len(st.session_state.alice_and_bob_matching_bases_str.replace('_', ''))

        st.text(f"Bob guessed correctly {guessed_bases_bob} times.")

        # Alice and Eve basis informtation
        if eve_present and 'alice_and_eve_matching_bases_str' not in st.session_state:

            matching_bases = generate_string(
                st.session_state.alice_bases_str, 
                st.session_state.eve_bases_str,
                mode='matching')
            
            st.session_state.alice_and_eve_matching_bases_str = matching_bases

        if eve_present:

            st.write(" ")

            st.markdown(f"""
            <div style="overflow-x: auto; white-space: pre; font-family: monospace; font-size: 13px; margin-bottom: 10px;">Alice's Bases      : {st.session_state.alice_bases_str}
            Eve's Bases        : {st.session_state.eve_bases_str}
            A-E Matching Bases : {st.session_state.alice_and_eve_matching_bases_str}
            </div>
            """, unsafe_allow_html=True)

            guessed_bases_eve = len(st.session_state.alice_and_eve_matching_bases_str.replace('_', ''))

            st.text(f"Eve guessed correctly {guessed_bases_eve} times.")

            st.write(" ")

            # Calcular y mostrar los qubits alterados por Eve
            altered_qubits = sum(
                1 for i in range(len(st.session_state.eve_bases_str))
                if st.session_state.eve_bases_str[i] != st.session_state.alice_bases_str[i] and
                st.session_state.bob_bases_str[i] == st.session_state.alice_bases_str[i]
            )

            lucky_qubits = sum(
                1 for i in range(len(st.session_state.eve_bases_str))
                if st.session_state.eve_bases_str[i] == st.session_state.alice_bases_str[i] and
                st.session_state.bob_bases_str[i] == st.session_state.alice_bases_str[i]
            )

            altered_qubits_string = generate_string(
                st.session_state.alice_and_bob_matching_bases_str,
                st.session_state.alice_and_eve_matching_bases_str,
                mode='error')
            
            lucky_qubits_string = generate_string(
                st.session_state.alice_and_bob_matching_bases_str,
                st.session_state.alice_and_eve_matching_bases_str,
                mode='lucky')

            st.markdown(f"""
            <div style="overflow-x: auto; white-space: pre; font-family: monospace; font-size: 13px; margin-bottom: 10px;">Error Bits         : {altered_qubits_string}
            Lucky Bits         : {lucky_qubits_string}
            </div>
            """, unsafe_allow_html=True)
            
            st.text(f"""
                    Eve altered {altered_qubits} qubits (Eve guessed wrong and Bob guessed right).
                    Eve got lucky {lucky_qubits} times (Eve guessed right and Bob guessed right).
                    """)


        if st.button("Discard bits"):

            st.session_state.section = 7


    # Sección 7: Sifted key
    if st.session_state.section >= 7:

        st.write("#### Sifted key")

        if 'discarded_bits_str' not in st.session_state: 
            
            discarded_bits = generate_string(
                st.session_state.alice_bases_str, 
                st.session_state.bob_bases_str,
                mode='unmatching')
        
            st.session_state.discarded_bits_str = discarded_bits

        if 'alice_sifted_key_str' not in st.session_state: 
            
            sifted_key = generate_string(
                st.session_state.alice_bits_str,
                st.session_state.alice_and_bob_matching_bases_str,
                mode='sifted_key')
        
            st.session_state.alice_sifted_key_str = sifted_key

        if 'bob_sifted_key_str' not in st.session_state: 
            
            sifted_key = generate_string(
                st.session_state.bob_bits_str,
                st.session_state.alice_and_bob_matching_bases_str,
                mode='sifted_key')
        
            st.session_state.bob_sifted_key_str = sifted_key
        
        st.markdown(f"""
        <div style="overflow-x: auto; white-space: pre; font-family: monospace; font-size: 1em; margin-bottom: 10px;">A-E Matching Bases : {st.session_state.alice_and_bob_matching_bases_str}
        Alice's Bits       : {st.session_state.alice_bits_str}
        Bob's Bits         : {st.session_state.bob_bits_str}
        Discarded Bits     : {st.session_state.discarded_bits_str}
        Alice's Sifted Key : {st.session_state.alice_sifted_key_str}
        Bob's Sifted Key   : {st.session_state.bob_sifted_key_str}
        </div>
        """, unsafe_allow_html=True)
        
        percentages = ["10%", "20%", "30%"]
        selected_percentage = st.radio("Bit fraction to reveal", percentages)

        if st.button("Reveal some bits"):

            st.session_state.section = 8


    # Sección 8: Security check
    if st.session_state.section >= 8:

        st.write('#### Security check')

        if 'revealed_bits_indexes' not in st.session_state:

            bob_sifted_key_indexes = [i for i, bit in enumerate(st.session_state.bob_sifted_key_str) 
                                    if bit != ' ']
            
            percentage = int(selected_percentage.strip('%'))

            sample_size = len(bob_sifted_key_indexes) * percentage // 100

            st.session_state.revealed_bits_indexes = np.random.choice(bob_sifted_key_indexes,
                                                                    sample_size,
                                                                    replace=False)
        
        if 'revealed_bits_str' not in st.session_state:

            # Mostrar los bits que se van a comparar
            st.session_state.revealed_bits_str = ''.join([
                st.session_state.bob_sifted_key_str[i] 
                if i in st.session_state.revealed_bits_indexes else ' ' 
                for i in range(len(st.session_state.bob_sifted_key_str))]
            )
        
        st.markdown(f"""
        <div style="overflow-x: auto; white-space: pre; font-family: monospace; font-size: 13px; margin-bottom: 10px;">Bob's Sifted Key : {st.session_state.bob_sifted_key_str}
        Revealed Bits    : {st.session_state.revealed_bits_str}
        </div>
        """, unsafe_allow_html=True)

        revealed_bits_length = len(st.session_state.revealed_bits_str.replace(' ', ''))

        p_d = 1 - (3/4) ** revealed_bits_length

        st.text(f"Probability of detecting Eve: {p_d:.2%}")

        if st.button("Compare bits"):

            st.session_state.section = 9


    # Sección 8: Compared bits
    if st.session_state.section >= 9:

        # Calcular la tasa de error cuántico de bits (QBER)
        if 'QBER' not in st.session_state:
            
            total_bits = len(st.session_state.revealed_bits_indexes)
            error_bits = sum(
                st.session_state.bob_sifted_key_str[i] != st.session_state.alice_sifted_key_str[i] 
                for i in st.session_state.revealed_bits_indexes)
            
            st.session_state.QBER = error_bits / total_bits if total_bits > 0 else 0

        matching_bits = generate_string(
            st.session_state.alice_bits_str, 
            st.session_state.revealed_bits_str,
            mode='mixed_matching')

        st.markdown(f"""
        <div style="overflow-x: auto; white-space: pre; font-family: monospace; font-size: 13px; margin-bottom: 10px;">Alice's Sifted Key : {st.session_state.alice_sifted_key_str}
        Revealed Bits      : {st.session_state.revealed_bits_str}
        Matching Bits      : {matching_bits}
        </div>
        """, unsafe_allow_html=True)

        st.text(f"The Quantum Bit Error Rate is {st.session_state.QBER:.2%}")

        # Definir el umbral (por ejemplo, 11% de tasa de error)
        qber_threshold = 0.11

        if st.session_state.QBER <= qber_threshold:
            
            st.success("No significant eavesdropping detected. The key is secure.")

        else:
            
            st.error("Discrepancies found. Potential eavesdropping detected.")

        # Mostrar botón de "Show resulting key" solo si no hay errores significativos
        if st.button("Show resulting key"):
            # Generar la Final Key y guardarla en el estado
            st.session_state.final_key_str = ''.join([
                bit if i not in st.session_state.revealed_bits_indexes else ' ' 
                for i, bit in enumerate(st.session_state.bob_sifted_key_str)])

            st.session_state.section = 10


    # Sección 10: Show resulting key
    if st.session_state.section >= 10:

        final_key_length = len(st.session_state.final_key_str.replace(' ', ''))

        # Mostrar la Sifted Key, los Revealed Bits y la Final Key
        st.markdown(f"""
        <div style="overflow-x: auto; white-space: pre; font-family: monospace; font-size: 13px; margin-bottom: 10px;">Bob's Sifted Key : {st.session_state.bob_sifted_key_str}
        Revealed Bits    : {st.session_state.revealed_bits_str}
        Final Key        : {st.session_state.final_key_str}
        </div>
        """, unsafe_allow_html=True)

        st.text(f"The final key length is {final_key_length} bits.")

        # Generar gráficos y obtener las rutas de las imágenes
        image_paths = visualize_statistics(
            bit_string_to_array(st.session_state.bob_bits_str),
            st.session_state.alice_sifted_key_str, 
            st.session_state.bob_sifted_key_str,
            st.session_state.revealed_bits_indexes,
            st.session_state.QBER
        )

        # Control deslizante para el carrusel
        image_index = st.slider("Simulation review", 0, len(image_paths) - 1, 0)

        # Mostrar la imagen seleccionada
        st.image(image_paths[image_index], use_column_width=True)


elif menu == "📚 About the protocol":

    st.markdown("""
    In 1984, **Charles Bennett** and **Gilles Brassard** developed the first quantum key distribution protocol, called BB84, which uses classical bit coding in qubits generated from photon polarization.

    The polarization of light describes the oscillation orientation of the electromagnetic field associated with its wave. In the case of linear polarization, this oscillation occurs in only one direction. Linear polarization states can be described by two bases: the rectilinear base, which includes horizontal and vertical orientations, and the diagonal base, which includes orientations rotated at 45° and 135°.
                
    A classical bit can be encoded in the polarization of a photon as shown in the Bit Encoding table. In the rectilinear basis, 0 is represented by horizontal polarization $(\ket{0})$, and 1 is represented by vertical polarization $(\ket{1})$. In the diagonal basis, 0 is represented with polarization at 45°, i.e. diagonal $(\ket{+})$, and 1 is represented with polarization at 135°, i.e. antidiagonal $(\ket{-})$.
    """)

    # Generar y obtener las figuras de Bloch desde la caché
    bloch_figures = generate_bloch_figures()

    # Crear una tabla en Streamlit
    st.write("#### Bit encoding")

    # Definir las columnas de la tabla
    col1, col2, col3 = st.columns([1, 1, 1])

    # Encabezados de la tabla
    with col1:
        st.write("**Bit value**")
    with col2:
        st.write("**Rectilinear basis (0)**")
    with col3:
        st.write("**Diagonal basis (1)**")

    # Primera fila: bit 0
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.write("0")
    with col2:
        st.pyplot(bloch_figures["|0⟩"])
        col2.markdown("$$|0⟩$$", unsafe_allow_html=True)
    with col3:
        st.pyplot(bloch_figures["|+⟩"])
        col3.markdown("$$|+⟩$$", unsafe_allow_html=True)

    # Segunda fila: bit 1
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.write("1")
    with col2:
        st.pyplot(bloch_figures["|1⟩"])
        col2.markdown("$$|1⟩$$", unsafe_allow_html=True)
    with col3:
        st.pyplot(bloch_figures["|−⟩"])
        col3.markdown("$$|−⟩$$", unsafe_allow_html=True)


    st.markdown("""
                
    The process of decoding a classical bit from the polarization state of a photon depends on the basis used during measurement. As shown in the Bit Decoding table, if the photon is measured in the rectilinear basis, a horizontally polarized state $(\ket{0})$ will yield a bit value of 0, while a vertically polarized state $(\ket{1})$ will yield a bit value of 1. However, if the photon is measured in the diagonal basis, the results are probabilistic.
                
    A photon in the diagonal state $(\ket{+})$, which corresponds to a 45° polarization, will be measured as a 0 in the diagonal basis. In contrast, the same photon measured in the rectilinear basis will have a 50% probability of being observed as either horizontally $(\ket{0})$ or vertically polarized $(\ket{1})$, leading to a random bit value of 0 or 1. Similarly, a photon in the antidiagonal state $(\ket{-})$, or 135° polarization, will yield a 1 when measured in the diagonal basis, but its measurement in the rectilinear basis will also result in a random bit value of 0 or 1.

    """)


    st.write("#### Bit decoding")

    # Definir las columnas de la tabla
    col1, col2, col3 = st.columns([1, 1, 1])

    # Encabezados de la tabla
    with col1:
        st.write("**Measured state**")
    with col2:
        st.write("**Rectilinear basis (|0⟩, |1⟩)**")
    with col3:
        st.write("**Diagonal basis (|+⟩, |−⟩)**")

    # Primera fila: estado medido |0⟩
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.write("$$|0⟩$$")
    with col2:
        st.markdown("$$|0⟩$$", unsafe_allow_html=True)
        st.markdown("Result: 0")
    with col3:
        st.markdown("$$\\frac{1}{\\sqrt{2}}|+⟩ + \\frac{1}{\\sqrt{2}}|−⟩$$", unsafe_allow_html=True)
        st.markdown("Result: Random (0 or 1)")

    # Segunda fila: estado medido |1⟩
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.write("$$|1⟩$$")
    with col2:
        st.markdown("$$|1⟩$$", unsafe_allow_html=True)
        st.markdown("Result: 1")
    with col3:
        st.markdown("$$\\frac{1}{\\sqrt{2}}|+⟩ - \\frac{1}{\\sqrt{2}}|−⟩$$", unsafe_allow_html=True)
        st.markdown("Result: Random (0 or 1)")

    # Tercera fila: estado medido |+⟩
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.write("$$|+⟩$$")
    with col2:
        st.markdown("$$\\frac{1}{\\sqrt{2}}|0⟩ + \\frac{1}{\\sqrt{2}}|1⟩$$", unsafe_allow_html=True)
        st.markdown("Result: Random (0 or 1)")
    with col3:
        st.markdown("$$|+⟩$$", unsafe_allow_html=True)
        st.markdown("Result: 0")

    # Cuarta fila: estado medido |−⟩
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.write("$$|−⟩$$")
    with col2:
        st.markdown("$$\\frac{1}{\\sqrt{2}}|0⟩ - \\frac{1}{\\sqrt{2}}|1⟩$$", unsafe_allow_html=True)
        st.markdown("Result: Random (0 or 1)")
    with col3:
        st.markdown("$$|−⟩$$", unsafe_allow_html=True)
        st.markdown("Result: 1")
