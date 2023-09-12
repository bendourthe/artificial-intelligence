___

<a href='https://www.dourthe-technologies.com'> <img src='img/Dourthe_Technologies_Headers.png' /></a>
___
<center><em>For more information, visit <a href='https://www.dourthe-technologies.com'> www.dourthe-technologies.com</a></em></center>

# Llama-2 Local Inference in Jupyter Noteboo

This notebook enables you to harness the power of the Llama-2 large language model, boasting an impressive 7 billion parameters, for text generation tasks. With this tool, you can define custom prompts and run Llama-2 locally on your own GPU, ensuring both privacy and control over your data.

## Features:

- **Local Execution**: Run Llama-2 on your personal GPU, avoiding the need for external servers or cloud services.

- **Custom Prompts**: Craft specific prompts to generate tailored responses from the model.

- **High Performance**: Leverage the full capacity of the 7B parameter Llama-2 model for natural language understanding and generation.

## Installation

Before using this tool, you need to set up the environment and access the Llama-2 model. Follow these steps for installation:

### Installation

To set up the environment for this project, follow these steps:

1. **Create a Project Directory**

   - Create a dedicated project directory where you want to organize all the project-related files and the Llama repository. You can do this in your terminal (Linux or VS Code), for example:
     ```
     mkdir llama-project
     cd llama-project
     ```

2. **Clone Meta AI Llama Repository**

   - Clone the Llama repository into your project directory:
     ```
     git clone https://github.com/facebookresearch/llama.git
     ```

   - Make sure that your current Jupyter notebook or script is located within this project directory for easier access to project files.

3. **Request Permission to Access Llama-2**

   - Go to [Meta AI Llama Downloads](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).
   - Fill in your information and select the `Llama 2 & Llama Chat` checkbox.
   - Accept the terms and conditions, then click `Accept and Continue`.
   - You will receive an email from Meta with a custom URL for accessing Llama 2.

4. **Download Models**

   - Change directory to the cloned `llama` repository within your project directory:
     ```
     cd llama
     ```
   - Run the following commands (Linux):
     ```
     chmod 755 download.sh
     ./download.sh
     ```

5. **Create a Virtual Environment**

   - If you are still in the `llama` directory, return to the project directory by going back one level:
     ```
     cd ../
     ```
   - Create a virtual environment using the following command (using Python 3.11 as an example):
     ```
     python3.11 -m venv llama-venv
     ```

6. **Install Requirements**

   - Activate your virtual environment:
     ```
     source llama-venv/bin/activate
     ```
   - Change directory back to the `llama` repository within your project directory:
     ```
     cd llama
     ```
   - Run the following command to install all the required Python packages in your virtual environment:
     ```
     pip install -e .
     ```

7. **Download 'complete_text.py'**

   - Download the 'complete_text.py' file from [your source] (replace with your source URL).
   - Move the downloaded 'complete_text.py' file to the 'llama' repository directory within your project.

Now that you have successfully set up your environment and organized your project directory, you can proceed to define model parameters, prompts, and run the Llama-2 model for text generation tasks.


## Usage

1. **Initialization**

   Before using the Llama-2 model, ensure you run the Initialization cell to import all the required libraries and define essential helper functions.

2. **Define Model Parameters**

   You can customize the behavior of the Llama-2 model by specifying various parameters:

   - **Model Name**: The name of the model used.
     ```python
     model_name = "llama-2-7b-chat"
     ```

   - **Llama Directory**: The path to the directory containing the cloned Llama repository.
     ```python
     llama_dir = "../llama"
     ```

   - **Complete Text Path**: The path to the complete text function (used to generate responses).
     ```python
     complete_text_path = f"{llama_dir}/complete_text.py"
     ```

   - **Checkpoint Directory**: The directory containing checkpoint files for the pretrained model.
     ```python
     ckpt_dir = f"{llama_dir}/{model_name}"
     ```

   - **Tokenizer Path**: The path to the tokenizer model used for text encoding and decoding.
     ```python
     tokenizer_path = f"{llama_dir}/tokenizer.model"
     ```

   - **Temperature**: A parameter controlling randomness in generation. Lower values make output less random.
     ```python
     temperature = 0.2
     ```

   - **Top-p Sampling**: A parameter controlling diversity in generation. Higher values result in more diverse output.
     ```python
     top_p = 0.9
     ```

   - **Maximum Sequence Length**: The maximum length of input prompts (in number of characters).
     ```python
     max_seq_len = 4096
     ```

   - **Maximum Generated Length**: The maximum length of generated sequences.
     ```python
     max_gen_len = 4096
     ```

   - **Maximum Batch Size**: The maximum batch size for generating sequences.
     ```python
     max_batch_size = 4
     ```

3. **Define Your Prompt**

   Define your specific prompt in the code, tailoring it to your text generation task.

4. **Run the Model**

   Execute the script to run the Llama-2 model with your defined prompt. The model will generate a response based on your input, leveraging the specified parameters.


## Troubleshooting Tips

If you encounter issues while setting up or running the Llama-2 model, consider these troubleshooting tips:

### 1. GPU and Nvidia Drivers

**Problem:** The code relies on a GPU with the right Nvidia drivers installed, including the CUDA Toolkit.

**Solution:** Ensure that your computer has a compatible GPU with up-to-date Nvidia drivers and the CUDA Toolkit installed. Verify that your GPU is recognized and accessible by the code.

### 2. GPU Memory Limitation

**Problem:** Your local GPU may not have enough memory to run the model with certain parameter settings.

**Solution:** If you encounter memory-related errors, consider reducing the `max_seq_len` and `max_gen_len` parameters to a more manageable size, like 512 and 256, as they have been known to work successfully on some configurations. Experiment with smaller values to find a balance between performance and available GPU memory.

### 3. Parameter Configuration

**Problem:** Incorrectly set parameters, including file and directory paths, can lead to errors.

**Solution:** Double-check all parameter settings, including file paths and directory locations. Ensure that they point to the correct files and directories required for the code to run successfully.

If you continue to experience issues, consult the official Llama-2 documentation or seek assistance from the support channels provided by the model's creators for more specific troubleshooting and guidance.

___

# Initialization


```python
# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #

import subprocess
import textwrap


# --------------------------------------------------------------------------- #
# Helper function
# --------------------------------------------------------------------------- #

def format_and_print_string(input_string, line_length=79):
    """
    Format and print a string with escaped characters to make it readable.

    This function takes an input string containing escaped characters such as '\\n' and '\\"'
    and formats it to replace those escape sequences with their actual representations.
    It then prints the formatted string with proper line breaks.

    Args:
        input_string (str): The input string with escaped characters to be formatted and printed.
        line_length (int): The maximum line length before wrapping. Defaults to 79.

    Returns:
        None: This function does not return a value; it prints the formatted and wrapped string.
    """

    # Replace escaped characters with their actual representations
    formatted_string = input_string.replace('\\n', '\n').replace('\\"', '"')

    # Wrap lines longer than line_length while preserving leading whitespace
    wrapped_lines = []
    for line in formatted_string.splitlines():
        wrapped_lines.extend(textwrap.wrap(line, width=line_length, subsequent_indent=''))

    # Join the wrapped lines and print the result
    wrapped_string = '\n'.join(wrapped_lines)
    print(wrapped_string)

```

# Model parameters


```python
# --------------------------------------------------------------------------- #
# Model parameters
# --------------------------------------------------------------------------- #

# The name of the model used
model_name = "llama-2-7b-chat"
# The path to the directory containing the llama cloned repository
llama_dir = "../llama"
# The path to the complete text function (used to generate response)
complete_text_path = f"{llama_dir}/complete_text.py"
# The directory containing checkpoint files for the pretrained model
ckpt_dir = f"{llama_dir}/{model_name}"
# The path to the tokenizer model used for text encoding/decoding
tokenizer_path= f"{llama_dir}/tokenizer.model"
# The temperature value for controlling randomness in generation
# (the lower the less random)
temperature = 0.2
# The top-p sampling parameter for controlling diversity in generation
top_p = 0.9
# The maximum sequence length for input prompts (in number of characters)
max_seq_len = 512
# The maximum length of generated sequences.
max_gen_len = 256
# The maximum batch size for generating sequences
max_batch_size = 4
```

# Prompt


```python
# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

# Define prompt used to generate response
prompt = 'Say Hi in 5 different languages.'
```

# Generate response
## Run model


```python
# --------------------------------------------------------------------------- #
# Run model and generate response
# --------------------------------------------------------------------------- #

# Define the command to run the script with specified arguments
command = [
    "torchrun",  # Replace with the actual command you need
    "--nproc_per_node",
    "1",
    f"{complete_text_path}",
    "--ckpt_dir",
    f"{ckpt_dir}",
    "--tokenizer_path",
    f"{tokenizer_path}",
    "--max_seq_len",
    f"{max_seq_len}",
    "--max_gen_len",
    f"{max_gen_len}",
    "--max_batch_size",
    f"{max_batch_size}",
    "--prompt",
    f"{prompt}"
]

# Execute the command and capture the output
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
stdout, stderr = process.communicate()
```

## Display response


```python
# Check for error and display model output
if process.returncode != 0:
    print(f"Error: {stderr}")
else:
    print("# --------------------------------------------------------------------------- #")
    print(f"# ORIGINAL PROMPT")
    print("# --------------------------------------------------------------------------- #")
    print("\n")
    print(f"{prompt}")
    print("\n\n")
    print("# --------------------------------------------------------------------------- #")
    print(f"# RESPONSE")
    print("# --------------------------------------------------------------------------- #")
    print("\n")
    # Select response from STDOUT
    escaped_string = stdout.split('[')[1].split(f"{prompt}")[1][4:-2]
    # Format and print model response
    format_and_print_string(escaped_string)

```

    # --------------------------------------------------------------------------- #
    # ORIGINAL PROMPT
    # --------------------------------------------------------------------------- #
    
    
    Say Hi in 5 different languages.
    
    
    
    # --------------------------------------------------------------------------- #
    # RESPONSE
    # --------------------------------------------------------------------------- #
    
    
    Saying "Hi" in different languages can be a fun and interesting way to connect
    with people from different cultures. Here are five ways to say "Hi" in
    different languages:
    1. Spanish: Hola (OH-lah)
    2. French: Bonjour (bone-JOOR)
    3. German: Hallo (HA-lo)
    4. Italian: Ciao (CHOW)
    5. Chinese: 你好 (nǐ hǎo) (Mandarin) or 您好 (nín hǎo) (Cantonese)
    Note: Pronunciation is approximate and may vary depending on the speaker and
    dialect."


#
