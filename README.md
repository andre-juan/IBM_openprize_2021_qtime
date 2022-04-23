# IBM Quantum Awards: Open Science Prize 2021/2022 - QTime solution

In this repository we present our solution for the IBM Quantum Awards: Open Science Prize 2021.

```
.
├── solution_description.pdf
├── final_versions/
│   ├── figs/ 
│   ├── results/ 
│   ├── results_other_members/ 
│   ├── troter_utils.py
│   ├── main_solution_notebook.ipynb
│   ├── classical_opt_larger_times.ipynb
│   ├── classical_opt_smaller_times_jobs_sent.ipynb
│   ├── classical_opt_negative_times_jobs_sent.ipynb
│   ├── final_results_analysis.ipynb
├── initial_versions/
└── README.md
```

______________

In the file `solution_description.pdf` you can find a detailed description of our approach for the solution. The text motivates and complements the code in the notebooks and main module.

______________

In the folder `final_versions` you find the final versions of the code produced for our solution. In what follows, we detail each file in this folder.

- `troter_utils.py`: this is the main module associated with our solution. In this file you can find the imports as well as all the functions written and used in the solution. All functions defined in this module are thoroughly commented, motivated in the solution description, and written in a way that their inner workings are clear. This module is imported and used in all the notebooks, in order to make them concise and straightforward. 

- `main_solution_notebook.ipynb` this is the main notebook for our solution. In the markdown cells, you can find essential aspects of what is presented in details in the `solution_description` pdf. As for the code, the input parameters were chosen as those which yielded the best result (highest mean fidelity in hardware execution), and all functions necessary to run our solution end-to-end were placed in the appropriate order. Notice, however, that this notebook has no output. This was intentionally done, so that the judges can directly run the code and verify our results. In the notebooks detailed below, we present the full code as well as the outputs, with which it is possible to identify the generation of the presented results. If one desires to re-run all of these notebooks, though, it is perfectly possible as well.

- `classical_opt_larger_times.ipynb`: in this notebook, the classical optimization of the variational circuit is performed, and the results are saved as parquet files (for the better parsing of data structures such as lists, which are in some columns of the dataframe). Notice: `pyarrow` (or `fastparquet`) is necessary for working with parquet files, so make sure to install this module for the code to be fully reproducible. In this first notebook, we use relatively large minimum times for each trotter step in the constrained optimization, as a first experiment. In the markdown cells of this notebook, the general rationale and structure of the experiments (which are replicated in the following notebooks) are thoroughly explained, so they are worth reading.

- `classical_opt_smaller_times_jobs_sent.ipynb`: this notebook is structurally similar to the previous one: the classical optimization is performed, and the results are saved. Beyond that, we also added an extra section `Hardware execution`, in which we submit the jobs for hardware execution. To better retrieve the results, we save a pickle file with a dictionary in the following format: `{"experiment_identification": [list of respective jobs' IDs]}` for each experiment setting. This is used to retrieve the jobs and assist the final results' analysis, which is done in a dedicated notebook, described below.

- `classical_opt_negative_times_jobs_sent.ipynb`: this is the same as the previous notebook, but here we allow for negative parameters (please see the solution description for details on the rationale).

- `final_results_analysis.ipynb`: in this notebook, the completed hardware execution jobs are retrieved, and the results are analyzed and concatenated with the noisy simulation results, producing the final results tables, as well as a stacked table with the results of all experiments, sorted by the highest hardware mean fidelity.

Within the `final_versions` folder you can also find other sub-folders:

- `figs/`: folder with plots of the target state fidelity as a function of time, for the full Hamiltonian evolution (time from 0 to pi). Some of these plots are referenced in the solution description pdf.

- `results/`: folder with aforementioned parquet and pickle results files. The files in this folder are the ones generated by the code in the respective notebooks, in the indicated dates. The cells' outputs (as well as information from nbxtensions' `Èxecute Time`) of all aforementioned notebooks were kept, so that the generation of such results may be identified. 

- `results_other_members`: some of the results in the main results table (available in the .pdf with the solution description) were obtained via jobs which were sent for execution by other members of the team, and produced by slightly different codes, yielding slightly differently-structured results files. For this reason, these files were kept in this folder, separated from the other ones in the `results/` folder, whose production is registered in the aforementioned notebook outputs.

______________

In the folder `initial_versions` you can find several notebook (.ipynb) and script (.py) Python files. These represent our first ideas and attempts in the direction of the solution. Some of these attempts are referenced in the solution description pdf, which is why we opted to include them here. That said, these files were not thoroughly documented, and the code therein should not be used for final evaluation of the solution.

______________

Members of the team QTime:

- André Juan Ferreira Martins

- Anton Simen Albino

- Askery Canabarro

- Rafael Chaves

- Rodrigo Pereira
