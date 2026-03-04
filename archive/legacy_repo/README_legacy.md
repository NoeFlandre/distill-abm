## README

### **1. Design of Experiments (DoE)**
This section describes the organization of the design of experiments, including the meaning of coded values used in the scripts.

#### **Experimental Factors & Coding Scheme**
| Code | Factor Description |
|------|--------------------|
| -1   | BERT (Summary Model) |
| 1    | BART (Summary Model) |
| -1   | GPT (LLM) |
| 1    | Claude (LLM) |
| -1   | No Role |
| 1    | Role Present |
| -1   | No Example |
| 1    | Example Present |
| -1   | No Insights |
| 1    | Insights Present |

The meaning of the different factors is as follows:

- **Summary**: This corresponds to the model being used to summarize information into our report. Its values are either BART or BERT.
- **LLM**: This corresponds to the LLM being used to generate an analysis of the simulation results. Its values are either GPT or Claude.
- **Role**: This corresponds to the presence of a given role within the prompt used for the LLM. Its values are Yes or No.
- **Example**: This corresponds to the presence of a given example within the prompt used for the LLM. Its values are Yes or No.
- **Insights**: This corresponds to the presence of a query to the LLM to provide insights based on the results within the prompt used for the LLM. Its values are Yes or No.

---

### **2. Models**
This section provides an overview of the models used in this project and where they can be accessed. Each model was chosen based on the availability of comprehensive documentation and source code, along with a coverage of distinct simulation topics.


- **MHPM**: A model that simulates megaherbivore extinction due to human predation. More details can be found [here](https://www.comses.net/codebases/d42eac84-ac76-43f7-8c14-5f5cc5fbf7f8/releases/1.1.0/).
- **Milk Adoption ABM**: A model that simulates milk consumption adoption trends in the UK. More details can be found [here](https://www.comses.net/codebases/2dcf78b7-2b00-4814-873e-4e6ab8faed8b/releases/1.0.0/).
- **RAGE Learning Extension**: A model that simulates pasture-livestock-household dynamics with learning mechanisms. More details can be found [here](https://www.comses.net/codebases/e103feef-2785-41e6-affb-8306c979e83c/releases/1.0.0/).

---

### **3. Folder Structure**

Below is an organized view of the project directory:

```
/project-root
│── /Code          # Scripts and source code
│── /Paper         # Research paper and documentation
│── /Presentations # Presentation materials
│── /Readings      # Relevant literature and references
│── /References    # Cited works and additional resources
│── /Results       # Output data and findings
│── README.md      # Project documentation
```

Please note that the evaluation was conducted from two different perspectives. Initially, we used human assessment, but due to insufficient interrater agreement, we shifted to qualitative evaluation using LLMs.
