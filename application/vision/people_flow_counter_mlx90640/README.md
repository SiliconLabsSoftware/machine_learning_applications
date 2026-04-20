# IR People Flow Counter

**This example is for demonstration purposes only and should not be considered a production-ready solution. Its purpose is to demonstrate technical viability and serve as a proof of concept.**

This example uses deep learning to detect objects with a low-resolution IR camera — people in this case — and determine bounding boxes around them. It also tracks movement and can be used to detect motion flows:

<p align="center">
<img src="./assets/animation.gif" alt="Demo Overview" width="200px"/>
</p>

The video above is an example of the output this demo provides. The output is the raw camera feed overlaid with the inference results, all produced by the board. The inference results are the bounding boxes (in red), from which centroids (white dots) can be derived and then tracked from frame to frame (white arrows).

The title shows information about people crossing the red line:

<ul>
    <li><b>L</b> and <b>R</b> are the number of people that have crossed from right to left and left to right respectively.</li>
    <li><b>T</b> is the number of people that have crossed from left to right and not yet crossed back.</li>
    <li><b>Present</b> is the number of people (the number of bounding boxes) currently detected by the model on the full image.</li>
</ul>

The main benefit of using deep learning to solve this problem is the ability to separate people who are close together and, more generally, separate objects that have overlapping IR signatures. This example also includes a way to gather new data, which is recommended to improve robustness and reduce the likelihood of false positives.

*The model training data is available upon request.*

## Setup

The following hardware is needed:

- MLX90640 110° IR camera ([link](https://www.sparkfun.com/products/14843))
- EFR32xG24 Dev Kit BRD2601B ([link](https://www.silabs.com/development-tools/wireless/efr32xg24-dev-kit?tab=overview)), referred to as "the device"
- A way to power the device, capable of supplying up to 30 mA at 3.3 V. Some options:
  - 2x AAA batteries in series connected to the Hirose DF13C-2P connector on the back of the xG24 dev kit
  - The USB connector in combination with a power source such as a power bank or laptop

The demo is intended to be set up like this:

<p align="center">
<img src="./assets/demo_design.png" alt="Demo Overview" width="400px"/>
</p>
<p align="center">
<img src="./assets/demo_design2.png" alt="Demo Overview" width="400px"/>
</p>

Considerations:

- The model is trained on data gathered at a height of approximately 3 meters, so the demo should be installed at a similar height.

An example setup using 2x AAA batteries looks like this:

<p align="center">
<img src="./assets/demo_example_qwiic.jpg" alt="Demo Overview" width="200px"/>
<img src="./assets/demo_example_battery.jpg" alt="Demo Overview" width="200px"/>
</p>

## Building and Flashing

This application has been validated with:

- Simplicity SDK (SiSDK) `2025.12.2`
- Silicon Labs AI/ML `2.2.1`
- Simplicity Studio `v6`

### Using Simplicity Studio

Simplicity Studio v6 is the recommended GUI workflow for SiSDK 2025.12.x.

Add this repository as a Simplicity SDK extension in Simplicity Studio v6, create the `IR People Flow Counter` project for `BRD2601B`, then build and flash it from the Studio workspace.

### Using the command line

A native CLI workflow is also possible.

From this application directory, generate the project directly from the `.slcp` file:

```sh
slc generate \
  -p people_flow_counter_mlx90640.slcp \
  --with brd2601b \
  --sdk-package-path "<path-to-simplicity-sdk>,<path-to-machine_learning_applications>,<path-to-aiml>" \
  -d people_flow_counter_mlx90640_brd2601b \
  --output-type vscode
```

Build the generated project:

```sh
cd people_flow_counter_mlx90640_brd2601b
cmake --workflow --preset project
```

Flash the generated firmware image using `commander`.

> Note: The exact generated output paths may vary depending on the generation mode and local tool versions.

> Note: The Python scripts under `misc/` are host-side tools for visualization, BLE transport, and data handling. They are part of the source repository and are not generated into the firmware project output.

## Connecting with Bluetooth Low Energy (BLE)

*Due to limitations in `bleak`, the Python BLE transport may show some instability depending on the OS. It has been tested on Windows and macOS, but reconnect behavior may vary. Forgetting and reconnecting the device can sometimes help.*

Start the BLE server on the host PC from the `misc/` directory:

```sh
python display_serial_ble_server.py
```

This starts a server that can connect to the device. The device is named `IR Device`. **Make sure the device is not already connected by another application or BLE client, otherwise it may not be discovered by the server.** If discovery fails, retry.

Install the Python dependencies with:

```sh
pip install -r requirements.txt
```

Once the server is connected, run the display client from the same `misc/` directory:

```sh
python display_serial_ble.py
```

A window should open showing the camera feed with the predictions.

## Displaying over serial

Displaying over serial may be useful for debugging because it is simpler and faster.

Before proceeding, reconfigure the board baud rate by following [this guide](https://community.silabs.com/s/article/wstk-virtual-com-port-baudrate-setting?language=en_US). The target baud rate is `921600`.

By default, the application uses BLE for output. To use serial output instead, set the `OUTPUT_OVER_BLE` define near the top of `people_counting.cc` to `false`.

After rebuilding and flashing, run the following script from the `misc/` directory:

```sh
python display_serial_local.py
```

A window should open showing the camera feed with the predictions.

## Notes

The display Python scripts accept other useful arguments that can be listed by executing the script with `-h` appended. Features such as data gathering and video recording may be enabled this way.

## Technical Details

### Performance

The table below displays performance numbers for the application, divided into separate processes. When running as a demo with raw camera feed output (referred to as debug), the application is significantly slower than it would be in a product configuration that only outputs people-flow and count data. The BLE measurements are slightly inaccurate due to the implementation prioritizing throughput rather than latency.

Values in **bold** are what one would expect in production.

| Operation                         | ms      | FPS      |
| --------------------------------- | ------- | -------- |
| Camera                            | 78      | -        |
| Preprocessing                     | 1       | -        |
| Inference                         | 35      | -        |
| Postprocessing                    | 1       | -        |
| Export results (Serial)           | 1       | -        |
| Export results (BLE)              | 12      | -        |
| Export debug information (Serial) | 33      | -        |
| Export debug information (BLE)    | 132     | -        |
| Total + Debug (Serial)            | 148     | 6.75     |
| Total + Debug (BLE)               | 262     | 3.81     |
| **Total (Serial)**                | **115** | **8.69** |
| **Total (BLE)**                   | **130** | **7.69** |

As for energy consumption, the camera uses the most energy, as shown in the following table:

| Component                        | mA        |
| -------------------------------- | --------- |
| Camera                           | 20        |
| Board (during inference, serial) | 4.51      |
| Board (during inference, BLE)    | 6.50      |
| **Total (Serial)**               | **24.51** |
| **Total (BLE)**                  | **26.50** |

### Model Architecture

The model is based on concepts from [this paper](https://arxiv.org/abs/2006.09214). The architecture itself differs from the paper, has only **11238 parameters**, and can be described using this simplified diagram:

<p align="center">
<img src="./assets/model_architecture.png" alt="Model Architecture Diagram" width="200px"/>
</p>

It has three resolution branches whose purpose is to view the input at multiple scales. This is intended to help the model learn differently sized structures in the respective branches.

### Deploying a model to the device

Deploying a model is simple and can be summarized in four steps:

1. Train a model using TensorFlow
2. Quantize the model (optional, but highly recommended)
3. Replace the `.tflite` model file in the `config/tflite/` folder
4. Re-generate, re-build, and re-flash the device

**A more detailed version is below:**

Assuming a model has been trained in TensorFlow, the first step is to quantize it. This can be done with the following code snippet:

```python
# Convert the model to the TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("model.tflite", "wb").write(tflite_model)

# Convert the model to the TensorFlow Lite format with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_dataset

tflite_model = converter.convert()

# Save the model to disk
open("model_quantized.tflite", "wb").write(tflite_model)

basic_model_size = os.path.getsize("model.tflite")
print(f"Basic model is {basic_model_size} bytes")
quantized_model_size = os.path.getsize("model_quantized.tflite")
print(f"Quantized model is {quantized_model_size} bytes")
difference = basic_model_size - quantized_model_size
print(f"Difference is {difference} bytes")
```

The representative dataset is a generator function used by the quantization algorithm. It provides samples intended to roughly represent the data distribution. More information is available [here](https://www.tensorflow.org/api_docs/python/tf/lite/RepresentativeDataset).

Once a quantized `.tflite` file has been produced, it can be copied into the `config/tflite/` folder to replace the existing model. The final step is to re-generate, re-build, and re-flash the device.

The new model is now deployed on the device.