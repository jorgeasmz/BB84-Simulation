# BB84-Simulation

This project implements an interactive simulation of the BB84 quantum key distribution protocol using Streamlit and Qiskit. It allows users to explore the protocol step-by-step, including scenarios with channel noise and the presence of an eavesdropper (Eve).

## Features

*   **Interactive Simulation:** A Streamlit-based user interface ([app.py](app.py)) that guides the user through the stages of the BB84 protocol.
*   **Scenario Selection:** Simulates the standard BB84 protocol, a scenario with a noisy quantum channel, and a scenario with an eavesdropper (Eve) intercepting the transmission.
*   **Step-by-Step Visualization:** Shows the generation of bits and bases by Alice, qubit preparation, base selection and measurement by Bob, and the classical post-processing to obtain the sifted key.
*   **Security Analysis:** Calculates and displays the Quantum Bit Error Rate (QBER) to detect the potential presence of Eve.
*   **Detailed Statistics:** Visualizes simulation statistics, such as histograms of measured bits, error rate, and key agreement ([functions.py](functions.py)).
*   **Protocol Explanation:** Includes an informative section within the application describing the fundamentals of the BB84 protocol, bit encoding, and decoding using quantum states ([app.py](app.py)).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <REPOSITORY-URL>
    cd BB84-Simulation
    ```
2.  **Install the dependencies:**
    Make sure you have Python 3.8+ installed. Then, install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the interactive simulation, use Streamlit:

```bash
streamlit run app.py