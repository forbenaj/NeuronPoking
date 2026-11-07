# Neuron Poking Paint Tool - An LLM MLP Hidden-Unit Steering Lab

Neuron Poking is a browser-based experiment for steering individual MLP hidden
units in Large Language Models using paint tools and knobs, specifically for Qwen2.5 0.5B Instruct and Gemma 3 1B IT. It runs the model locally with WebLLM and WebGPU.

## Run locally

Use a browser with WebGPU support (modern Chromium) and serve the files over HTTP.

From this directory, run:

```powershell
py -m http.server 8000
```

Then open <http://localhost:8000/index.html>, choose a model in the top
bar, and select **Load model**. The status bar clearly indicates when no model
is loaded and reports loading progress. This loads the model for both **Chat** and
**Tester**, which open from the bottom-corner tabs. Each model uses pre-converted
MLC weights and a matching precompiled WebLLM library, so compiling the source
model is not required.
The first load downloads model files from Hugging Face and the WebLLM runtime
from public CDNs; it can take a while and requires storage and enough GPU memory.
Subsequent loads may use the browser's cache. Select **Unload model** to release
the current model, clear the chat, and enable model selection again.

## Use

- Select Qwen or Gemma model before loading. Layer and neuron counts update for the
  selected model. The grid is fixed at 96 × 72 for Gemma and 76 × 64 for Qwen.
- Select a layer with the numbered buttons or the `[` and `]` keys.
- In pixel view, you can paint activations directly over the neurons with the brush,
  paint bucket, and rectangle tools. Choose a tool with its icon button; the
  selected tool is highlighted.
  Choose brush size and activation intensity; the brush outline previews its square footprint.
  Left-click and drag to draw, Right-click and drag to erase.
  Red represents negative activation and green represents positive activation.
- The value applies a gain of `1 + 4 × value` to that neuron's MLP activation.
- Use **Undo**/**Redo**, **Ctrl+Z**/**Ctrl+Y**, or **Ctrl+Shift+Z** to step
  through neuron edits and resets.
- Toggle **Knob mode** on the main screen to switch between knob and pixel
  interactions; it is off by default.
- In knob view, drag a neuron knob vertically to change its value between the configured
  negative and positive maximum; hold **Shift** for finer adjustments.
  Double-click a knob to reset it.
- The grid fits the available section at 100% zoom. Use **Alt + mouse wheel**
  anywhere in the section, or the **+** and **−** controls over the canvas, to
  zoom. The zoom controls remain visible while scrolling the canvas.
  **Reset** returns to the fit view.
- Send a message to generate text. Use **Reset layer** or **Reset all** to clear
  steering values.
- Tester is open by default. Use the bottom-corner **Tester** and **Chat** tabs
  to show or hide their panels; the active tab is highlighted.
- Chat's **max tokens** field is limited to 1–2048; blank or invalid input uses
  the default of 256.
- **Tester** runs a short inference after steering changes. It is on by default;
  edit its prompt or switch it off in the **Tester** window.
- **Settings** just has controls for the maximum activation for now, which defaults to 1 and can be increased up to 20.

Steering values and shared view settings are saved in this browser's local storage.
**Clear** resets the chat, not the steering settings.

## Implementation notes

Model IDs, MLC repositories, compiled libraries, and steering dimensions are
configured in `main.js`. The steering hook accesses WebLLM's internal TVM tensor
cache and edits quantized MLP projection scales in GPU memory. This depends on
WebLLM internals and model tensor layouts, so treat steering as experimental:
changes to the model artifacts or WebLLM version may require updating the hook.
Steering values are applied immediately before generation and the model's chat
and KV cache are reset for each generation.
