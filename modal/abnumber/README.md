# **Silica Modal**

This directory contains the code for the **abnumber** modal function, which helps in identifying complementarity-determining regions (CDRs) in sequences using various numbering schemes.

## **Prerequisites**

- [Anaconda](https://www.anaconda.com/products/individual) installed

## **Setup Instructions**

1. **Install Anaconda**  
   Follow the instructions provided on the [Anaconda website](https://www.anaconda.com/products/individual) to install Anaconda.

2. **Clone the Repository**  
   Clone the repository to your local machine by running the following command:

    ```bash
    git clone https://github.com/your-username/silica-modal.git
    ```

3. **Navigate to the Project Directory**  
   Change into the project directory:

    ```bash
    cd /Users/tompritsky/projects/silica/silica/modal/abnumber
    ```

4. **Create a Conda Environment**  
   Create a new Conda environment using the provided `environment.yml` file:

    ```bash
    conda env create -f environment.yml -n silica-modal
    ```

5. **Activate the Conda Environment**  
   Activate the newly created environment:

    ```bash
    conda activate silica-modal
    ```

## **Usage**

To run the `modal` command-line tool, use the following command:

```bash
modal run FindCDRs.py --scheme "imgt" --seq "test_modal_seqs.csv"

```

### **Options**:

- **`--scheme`**: Specify the numbering scheme to use for identifying CDRs. Choose one of the following:
  - `imgt`: Based on the IMGT numbering system.
  - `chothia`: Follows the Chothia CDR definitions.
  - `kabat`: Utilizes the Kabat numbering system for CDR identification.
  - `aho`: Uses the AHO scheme for structural annotation.

- **`--seq`**: Specify the input sequence(s). Two formats are supported:
  - A **single sequence** provided as a string for individual processing. Example:
    ```bash
    modal run FindCDRs.py --scheme "imgt" --seq "QVQLVQSGGGLVQP"
    ```
  - A **CSV file** with sequences listed in the 'sequence' column for batch processing. Outputs 'results.csv', a copy of the original with an additional column for cdr positions. Example:
    ```bash
    modal run FindCDRs.py --scheme "imgt" --seq "test_modal_seqs.csv"
    ```

