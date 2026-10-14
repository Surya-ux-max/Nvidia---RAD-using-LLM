# Notebook 0: JupyterLab Environment & Resource Management

Documentation for [00_jupyterlab.ipynb](file:///d:/Nvidia-RADU-LLM/00_jupyterlab.ipynb) from the **NVIDIA Deep Learning Institute (DLI): Rapid Application Development using Large Language Models**.

---

## 1. Overview & Core Concepts

This notebook serves as the operational onboarding guide to the interactive JupyterLab environment used throughout the NVIDIA Large Language Model (LLM) course.

### Key Concepts Covered:
- **JupyterLab Interface Architecture**:
  - **Menu Bar**: Controls file operations, kernel lifecycle, cell execution, and view options.
  - **Left Sidebar**: Contains the file browser, running sessions/kernels list, Git extensions, and command palette.
  - **Main Work Area**: Tabbed interface housing active notebooks, terminal sessions, and file editors.
- **Cell Structure & Execution**:
  - **Markdown Cells**: Narrative documentation formatted in Markdown and HTML.
  - **Code Cells**: Executable Python code blocks with standard output/error streams displayed directly beneath.
  - **Execution Shortcut**: `Shift + Enter` executes the currently focused cell and advances to the next cell.
- **GPU Memory & VRAM Management**:
  - LLM workloads allocate substantial VRAM on NVIDIA GPUs for model weights, KV caches, and activation tensors.
  - If a kernel crashes, raises an Out-Of-Memory (OOM) error, or finishes an experiment, VRAM remains occupied until the kernel process is explicitly terminated or restarted.

---

## 2. Code Walkthrough & Expected Outputs

### Exercise 1: Basic Cell Execution

#### Code:
```python
# Highlight this cell and click [Shift+Enter] to execute
print('This is just a simple print statement')
```

#### Explanation:
- Validates that the active Python kernel (`Python 3 (ipykernel)`) is responsive.
- Tests standard output stream (`stdout`) rendering directly under the cell.

#### Expected Output:
```text
This is just a simple print statement
```

---

### Exercise 2: Programmatic Kernel Shutdown & GPU Memory Cleanup

#### Code:
```python
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

#### Explanation:
- `import IPython`: Imports the core IPython framework.
- `IPython.Application.instance()`: Retrieves the singleton instance of the current running IPython notebook application.
- `app.kernel.do_shutdown(restart=True)`: Sends a shutdown signal to the underlying kernel process.
  - Passing `True` instructs the kernel manager to restart immediately after shutdown.
  - All global variables, allocated CUDA device contexts, and GPU VRAM tensors are immediately freed back to the operating system.

#### Expected Output / Behavior:
- In the JupyterLab UI, a notification briefly appears indicating **"Kernel restarting"** followed by **"Kernel ready"**.
- Output prompt counters (e.g., `In [1]:`) reset to blank `In [ ]:`.
- Output dictionary:
```json
{'status': 'ok', 'restart': True}
```

---

## 3. GPU Memory Clearing Strategies

When developing LLM applications, GPU state corruption or CUDA Out of Memory (`torch.cuda.OutOfMemoryError`) can occur. The guide establishes three remediation strategies:

| Strategy | Action Type | Procedure | Impact |
| :--- | :--- | :--- | :--- |
| **1. UI Soft Reset** | Soft Reset | Click the **⟳ (Restart Kernel)** icon in the notebook toolbar above the cells. | Clears Python scope and frees CUDA allocations while keeping the tab active. |
| **2. Left Sidebar Hard Stop** | Hard Stop | Click the **Running Terminals and Kernels** icon (circle containing a square) in the left panel, find the kernel, and click **Shut Down**. | Completely kills the kernel process and halts any background runaway tasks. |
| **3. Programmatic Reset** | Programmatic Soft Reset | Run the `app.kernel.do_shutdown(True)` snippet placed at the end of each lab notebook. | Automatically resets environment state before transitioning to the next notebook. |

---

## 4. Best Practices for Subsequent Notebooks
1. **Always clean up VRAM between notebooks**: Never keep multiple LLM-bearing notebooks running concurrently unless explicitly designed for multi-model serving.
2. **Re-render Markdown cells**: If a Markdown cell is accidentally double-clicked into editing mode, press `Shift + Enter` to re-render it.
3. **Verify Kernel Status**: Monitor the kernel status circle in the top right corner:
   - ○ (Empty circle): Idle / Ready
   - ⬤ (Filled circle): Busy / Executing
