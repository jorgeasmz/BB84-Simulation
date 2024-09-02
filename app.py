from PIL import Image
from functions import *

import time
import streamlit as st
import numpy as np


# Título de la aplicación
st.title('BB84 Protocol')

st.sidebar.markdown("<h2>BB84 Protocol</h2>", unsafe_allow_html=True)
st.sidebar.write("Explore a guided simulation of the BB84 protocol.")

# Menú en la barra lateral
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
    noise_percentage = 0
    eve_present = False

    if scenario == 'Noisy channel':
        noisy_channel = True

        noise_percentage = st.slider(
        "Amount of noise to simulate (%)",
        min_value=0,
        max_value=30,
        value=10,
        step=1,
        format="%.1f%%")

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
            
            bob_bits, _ = quantum_transmission(n, alice_bits, alice_bases, bob_bases, eve_bases=eve_bases, eve_present=eve_present, noisy_channel=noisy_channel, error_rate=noise_percentage)

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
        
        st.text(f"""
        Alice's Bases      : {st.session_state.alice_bases_str}
        Bob's Bases        : {st.session_state.bob_bases_str}
        A-B Matching Bases : {st.session_state.alice_and_bob_matching_bases_str}
        """)

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

            st.text(f"""
            Alice's Bases      : {st.session_state.alice_bases_str}
            Eve's Bases        : {st.session_state.eve_bases_str}
            A-E Matching Bases : {st.session_state.alice_and_eve_matching_bases_str}
            """)

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

            st.text(f"""
                    Error Bits         : {altered_qubits_string}
                    Lucky Bits         : {lucky_qubits_string}
                    """)
            
            st.text(f"""
                    Eve altered {altered_qubits} qubits (Eve guessed wrong and Bob guessed right).
                    Eve got lucky {lucky_qubits} times (Eve guessed right and Bob guessed right).
                    """)


        if st.button("Discard bits"):

            st.session_state.section = 7


    # Sección 7: Sifted key
    if st.session_state.section >= 7:

        st.write('#### Sifted key')

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
        

        st.text(f"""
        A-E Matching Bases : {st.session_state.alice_and_bob_matching_bases_str}
        Alice's Bits       : {st.session_state.alice_bits_str}
        Bob's Bits         : {st.session_state.bob_bits_str}
        Discarded Bits     : {st.session_state.discarded_bits_str}
        Alice's Sifted Key : {st.session_state.alice_sifted_key_str}
        Bob's Sifted Key   : {st.session_state.bob_sifted_key_str}
        """)
        
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
        
        st.text(f"""
        Bob's Sifted Key : {st.session_state.bob_sifted_key_str}
        Revealed Bits    : {st.session_state.revealed_bits_str}
        """)

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
        
        st.text(f"""
        Alice's Sifted Key : {st.session_state.alice_sifted_key_str}
        Revealed Bits      : {st.session_state.revealed_bits_str}
        Matching Bits      : {matching_bits}
        """)

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
        st.text(f"""
        Bob's Sifted Key : {st.session_state.bob_sifted_key_str}
        Revealed Bits    : {st.session_state.revealed_bits_str}
        Final Key        : {st.session_state.final_key_str}
        """)

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

    # Crear una tabla en Streamlit
    st.write("#### Bit encoding")

    bloch_0 = Image.open('images/bloch_0.png')
    bloch_1 = Image.open('images/bloch_1.png')
    bloch_p = Image.open('images/bloch_p.png')
    bloch_m = Image.open('images/bloch_m.png')

    table_html = """
    <style>
        .table-responsive {{
            width: 100%;
            max-width: 100%;
            overflow-x: auto;
            display: block;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 8px;
            text-align: center;
            border: 1px solid #ddd;
        }}
        img {{
            width: 125px;
            min-width: 100px;
            height: auto;
        }}
    </style>
    <div class="table-responsive">
        <table>
            <thead>
                <tr>
                    <th>Bit value</th>
                    <th>Rectilinear basis (0)</th>
                    <th>Diagonal basis (1)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>0</td>
                    <td><img src="data:image/png;base64,{img_0}" alt="|0⟩" /><br>|0⟩</td>
                    <td><img src="data:image/png;base64,{img_plus}" alt="|+⟩" /><br>|+⟩</td>
                </tr>
                <tr>
                    <td>1</td>
                    <td><img src="data:image/png;base64,{img_1}" alt="|1⟩" /><br>|1⟩</td>
                    <td><img src="data:image/png;base64,{img_minus}" alt="|−⟩" /><br>|−⟩</td>
                </tr>
            </tbody>
        </table>
    </div>
    """

    # Convertir las imágenes cargadas a base64
    img_0_base64 = pil_to_base64(bloch_0)
    img_1_base64 = pil_to_base64(bloch_1)
    img_plus_base64 = pil_to_base64(bloch_p)
    img_minus_base64 = pil_to_base64(bloch_m)

    table_html = table_html.format(
    img_0=img_0_base64,
    img_1=img_1_base64,
    img_plus=img_plus_base64,
    img_minus=img_minus_base64
    )

    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("""
                
    The process of decoding a classical bit from the polarization state of a photon depends on the basis used during measurement. As shown in the Bit Decoding table, if the photon is measured in the rectilinear basis, a horizontally polarized state $(\ket{0})$ will yield a bit value of 0, while a vertically polarized state $(\ket{1})$ will yield a bit value of 1. However, if the photon is measured in the diagonal basis, the results are probabilistic.
                
    A photon in the diagonal state $(\ket{+})$, which corresponds to a 45° polarization, will be measured as a 0 in the diagonal basis. In contrast, the same photon measured in the rectilinear basis will have a 50% probability of being observed as either horizontally $(\ket{0})$ or vertically polarized $(\ket{1})$, leading to a random bit value of 0 or 1. Similarly, a photon in the antidiagonal state $(\ket{-})$, or 135° polarization, will yield a 1 when measured in the diagonal basis, but its measurement in the rectilinear basis will also result in a random bit value of 0 or 1.

    """)

    st.write("#### Bit decoding")

    table_html = """
    <style>
        .table-responsive {{
            width: 100px;
            max-width: 100%;
            min-width: 400px;
            overflow-x: auto;
            display: block;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 8px;
            text-align: center;
            border: 1px solid #ddd;
        }}
    </style>
    <div class="table-responsive">
        <table>
            <thead>
                <tr>
                    <th>Measured state</th>
                    <th>Rectilinear basis (|0⟩, |1⟩)</th>
                    <th>Diagonal basis (|+⟩, |−⟩)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>|0⟩</td>
                    <td>
                        |0⟩<br/>
                        Result: 0
                    </td>
                    <td>
                        1/√2 [|+⟩ + |−⟩]<br/>
                        Result: Random (0 or 1)
                    </td>
                </tr>
                <tr>
                    <td>|1⟩</td>
                    <td>
                        |1⟩<br/>
                        Result: 1
                    </td>
                    <td>
                        1/√2 [|+⟩ - |−⟩]<br/>
                        Result: Random (0 or 1)
                    </td>
                </tr>
                <tr>
                    <td>|+⟩</td>
                    <td>
                        1/√2 [|0⟩ + |1⟩]<br/>
                        Result: Random (0 or 1)
                    </td>
                    <td>
                        |+⟩<br/>
                        Result: 0
                    </td>
                </tr>
                <tr>
                    <td>|−⟩</td>
                    <td>
                        1/√2 [|0⟩ - |1⟩]<br/>
                        Result: Random (0 or 1)
                    </td>
                    <td>
                        |−⟩<br/>
                        Result: 1
                    </td>
                </tr>
            </tbody>
        </table>
    </div>
    """

    st.markdown(table_html, unsafe_allow_html=True)

    st.write("#### Steps of the protocol")

    st.markdown("""
                
    The protocol consists of two stages: The first is quantum transmission, in which Alice and Bob prepare, send, and measure quantum states. The second stage is classical postprocessing. During this stage, Alice and Bob communicate only through the classical channel to convert the previously obtained bit sequences into secure keys.
                
    """)

    st.write("##### Quantum transmission")

    st.markdown("""
                
    1. Alice chooses a string of $n$ random classical bits $X_1, ..., X_n$.
    2. Alice selects a random sequence of polarization bases, choosing between the rectilinear base $(0)$ or the diagonal base $(1)$. These are conjugate bases, meaning that a measurement in one of the bases provides no information about a bit encoded in the other base.
    3. Alice encodes her bit sequence into a sequence of polarized photons according to the selected bases, as shown in the Bit Encoding table.
    4. Bob receives the photons and randomly decides (independently of Alice) for each photon whether to measure it in the rectilinear base or the diagonal base. At this point, both Alice and Bob have a sequence of classical bits, denoted as $X = (X_1, ..., X_n)$ for Alice and $Y = (Y_1, ..., Y_n)$ for Bob.
         
    """)

    st.write("##### Classical post-processing")

    st.markdown("""
    
    5. Bob makes public the information about the bases he used to measure the photons sent by Alice.
    6. Alice compares these bases with those she used in the preparation process and informs Bob which of his choices matched correctly. Then, both discard all bits where the encoding and measurement bases do not match.
    7. Alice and Bob calculate the error rate in the quantum channel, i.e., the proportion of positions where $X_i$ and $Y_i$ do not match. To do this, Bob randomly reveals some bits of his key. If Eve has not interfered, these bits should match Alice's, and she verifies them. If the error rate is too high, this suggests the presence of Eve, and Alice and Bob abort the protocol. The bits revealed in this step are discarded.
    8. The remaining bits form the shared key.

    """)
    