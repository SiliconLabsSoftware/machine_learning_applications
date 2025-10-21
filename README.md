# Silicon Labs Machine Learning Applications

Silicon Labs provides integrated hardware, software and development tools to help you quickly create secure, intelligent devices suitable for both industrial and commercial use cases. Our development platform has first class support for embedded machine learning (TinyML) model inference, backed by the [Tensorflow Lite for Microcontrollers (TFLM)](https://www.tensorflow.org/lite/microcontrollers) framework. We offer devices such as the [EFR32xG24](https://www.silabs.com/wireless/zigbee/efr32mg24-series-2-socs) dev kit that have hardware accelerators specifically built for [high-performant and energy efficient](https://mlcommons.org/en/inference-tiny-07/) AI/ML edge computing.

This repository contains a collection of embedded applications that leverage ML. You can use these to program your own Silicon Labs device, or as a starting point to develop your own TinyML application.

Feel free to open an issue if you have any questions or encounter problems, but take note that unless otherwise stated, all examples are considered to be EXPERIMENTAL QUALITY. The provided code has not been formally tested and is provided as-is. It is not suitable for production environments. In addition, there may be no bug maintenance planned for these resources. Silicon Labs may update the repository from time to time.

## Table of Contents  <!-- omit from toc -->

- [Silicon Labs Machine Learning Applications](#silicon-labs-machine-learning-applications)
  - [About](#about)
  - [Building and running](#building-and-running)
    - [Build container](#build-container)
      - [Build container: Installing the prerequisites](#build-container-installing-the-prerequisites)
    - [Build container: Generating, building and flashing an application](#build-container-generating-building-and-flashing-an-application)
    - [Command line tools](#command-line-tools)
      - [Command line tools: Installing the prerequisites](#command-line-tools-installing-the-prerequisites)
      - [Command line tools: Generating, building and flashing an application](#command-line-tools-generating-building-and-flashing-an-application)
    - [Simplicity Studio](#simplicity-studio)
      - [Simplicity Studio: Adding an external repository](#simplicity-studio-adding-an-external-repository)
      - [Simplicity Studio: Flashing prebuilt demos](#simplicity-studio-flashing-prebuilt-demos)
      - [Simplicity Studio: Generating, building and flashing an application](#simplicity-studio-generating-building-and-flashing-an-application)
  - [Testing](#testing)
    - [Testing: Using build container](#testing-using-build-container)
    - [Testing: Using command line tools](#testing-using-command-line-tools)
      - [Natively: Running the tests](#natively-running-the-tests)
  - [License](#license)

___

## About

The repository is organized by use case category. All applications are self contained, and include their own documentation.

Within an application's directory you will generally find,

- Source code for training the ML model and exporting it to Tensorflow Lite or a trained model artifact
- Documentation on model training and usage

## Building and running

There are multiple demo applications and project templates in this repository. We provide a Dockerfile which defines a build container for building and testing the code. Instructions for building natively using command line tools or Simplicity Studio are located further down.

### Build container

The Dockerfile at `build/Dockerfile` provides tooling for running containerized application builds and tests.

To build and run an application for your board using the build container, you will need to:

1. Install the prerequisites
2. Use the build container to generate a project for your board using Silicon Labs Configurator and compile it using Make
3. Copy the compiled application binaries to your host machine
4. Use Simplicity Commander on your host machine to flash the compiled application onto your device

#### Build container: Installing the prerequisites

To install the prerequisites and build the build container image,

1. To compile the code you'll need a Docker-compatible CLI, e.g. [Docker](https://www.docker.com/) or [Rancher (with the `dockerd` engine)](https://rancherdesktop.io/).
2. To flash binaries onto your device, you'll need `commander` ([Simplicity Commander](https://www.silabs.com/developers/mcu-programming-options#programming))

3. After installing the tools, make sure that they are available in your `PATH`.
4. Then, clone this repository.

    ```sh
    git clone https://github.com/SiliconLabs/machine_learning_applications
    ```

5. Lastly, build the container image using `docker`.

    ```sh
    # Navigate to the repository
    cd machine_learning_applications
    # Build the container image
    docker build -t mla-builder -f build/Dockerfile .
    ```

    If on `aarch64`, use the following command instead:

    ```sh
    DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t mla-builder -f build Dockerfile .
    ```

### Build container: Generating, building and flashing an application

You can use the build container to generate and compile a project based on project templates provided by this repository. The example below demonstrates this for the `sensory_wakeupword_series_2` application, but a similar procedure can be used for other applications.

1. On your host machine, run the build container and bind a local directory to a directory on the container

    ```sh
    # Start bash in the build container, with /tmp bound to your host machine's Downloads folder
    docker run -v $HOME/downloads:/tmp -it mla-builder bash
    ```

2. In the build container, generate an application for a specific Silicon Labs device and compile it.

    ```sh
    # Navigate to the application you want to build
    cd application/voice/sensory_wakeupword/app
    # Generate the application for the EFR32xG24 DevKit - BRD2601B
    slc generate sensory_wakeupword_series_2.slcp --with brd2601b -d target/brd2601b
    # Compile the application
    cd target/brd2601b
    make -f sensory_wakeupword_series_2.Makefile -j
    ```

3. In the build container, copy the compiled application binaries to your host machine

    ```sh
    # Copy application binaries to the host machine Downloads folder
    cp -r build /tmp/sensory_wakeupword
    ```

4. On your host machine, use `commander` to flash the application binaries onto your device,

    ```sh
    # Assuming you've connected a EFR32xG24 Dev Kit to your machine over USB,
    commander flash ~/Downloads/sensory_wakeupword/debug/sensory_wakeupword_series_2.s37
    # Note: If you encounter issues when flashing, try running `commander device recover` first.
    ```

### Command line tools

To build and run an application for your board using the command line tools, you will need to:

1. Install the prerequisites
2. Generate a project for your board using Silicon Labs Configurator
3. Build the project using Make and flash it onto your device using Simplicity Commander

#### Command line tools: Installing the prerequisites

1. To flash binaries onto your device, you'll need

   - `commander` ([Simplicity Commander](https://www.silabs.com/developers/mcu-programming-options#programming))

2. To generate and build the project, you'll need

   - `make` (for Windows we recommend using WSL or Cygwin)
   - `arm-eabi-none` GNU toolchain ([GNU ARM Embedded Toolchain](https://developer.arm.com/downloads/-/gnu-rm), version 10.3-2021.10 or 10.3-2021.07)
   - `slc` ([Silicon Labs Configurator](https://www.silabs.com/documents/public/user-guides/ug520-software-project-generation-configuration-with-slc-cli.pdf), installation instructions can be found under Section 2 in the provided link)

3. After installing the tools, make sure that they are available in your `PATH`.

4. Then, download `Gecko SDK >= v4.1`. You can obtain this from the Github repository [SiliconLabs/gecko_sdk](https://github.com/SiliconLabs/gecko_sdk).

    ```sh
    git clone https://github.com/SiliconLabs/gecko_sdk
    cd gecko_sdk
    git checkout v4.1.0
    ```

5. After this, configure `slc` to use the Gecko SDK and ARM GNU Embedded Toolchain you downloaded,

    ```sh
    slc configuration --sdk=<path/to/gecko_sdk>
    slc configuration --gcc-toolchain=<path/to/arm-gnu-gcc-toolchain>
    slc signature trust --sdk=<path/to/gecko_sdk>
    ```

6. Lastly, clone this repository.

    ```sh
    git clone https://github.com/SiliconLabs/machine_learning_applications
    ```

#### Command line tools: Generating, building and flashing an application

You can use the CLI tools to generate and compile a project based on project templates provided by this repository as follows. The example below demonstrates this for the `sensory_wakeupword_series_2` application, but a similar procedure can be used for other applications.

1. Generate a project targeting your device,

    ```sh
    # Navigate to the application
    cd application/voice/sensory_wakeupword
    # Then generate a project from the template
    slc generate app/sensory_wakeupword_series_2.slcp --with brd2601b -d target/brd2601b
    ```

2. Build the project using `make`,

    ```sh
    # Starting from voice/sensory_wakeupword
    cd target/brd2601b
    make -f sensory_wakeupword_series_2.Makefile -j
    ```

3. The compiled code can then be found under `target/brd2601b/build/debug`. You can flash the firmware onto your device with `commander` using the following command,

    ```sh
    # Starting from voice/sensory_wakeupword
    # Assuming you've connected a EFR32xG24 Dev Kit to your machine over USB,
    commander flash bin/efr32xg24_dev_kit/sensory_wakeupword_series_2.s37
    # Note: If you encounter issues when flashing, try running `commander device recover` first.
    ```

### Simplicity Studio

To build and run demos for your board using Simplicity Studio, you will need to:

1. Add this Git repository as a Gecko SDK extension in Simplicity Studio
2. Create a new project for your board using the Simplicity Studio Launcher
3. Build the code and flash it onto your device using the Simplicity Studio IDE

#### Simplicity Studio: Adding an external repository

Simplicity Studio supports adding Gecko SDK extensions which provide new project templates, prebuilt demos and software components. To add this repository as an SDK extension,

1. Download the code, either by
   - Cloning the repository using `git`:

      ```sh
      git clone https://github.com/SiliconLabs/machine_learning_applications.git
      ```

   - Or navigating to <https://github.com/SiliconLabs/machine_learning_applications.git>, opening the green `Code` dropdown, selecting `Download ZIP`, and extracting the contents.
2. Open Simplicity Studio
3. Open the Preferences view in Simplicity Studio (the cogwheel at the top of the window)
4. Select `Simplicity Studio`→`SDKs`
5. Select your Gecko SDK Suite, then click on `Add Extension...`.
6. In the menu, click on `Browse` and point to the root directory of the downloaded repository. Select `Machine Learning Applications` and click on `OK` to add the extension, then click on `Trust` to trust the contents.
7. Click `Apply and Close` to exit the preferences menu.

*Note: If `Simplicity Studio`→`SDKs` doesn't show up, you might be missing the 32-bit and Wireless MCUs package for Studio. You can install this by opening the Install view (the arrow at the top of the window) and selecting `Install by technology type`.*

#### Simplicity Studio: Flashing prebuilt demos

Some applications in this repository include prebuilt demo binaries which can be flashed onto your device without setting up a local project. To flash the prebuilt demo onto your device,

1. Open the Simplicity Studio Launcher (the rocket button in the top right corner of the window)
2. Connect your device (e.g. an EFR32xG24 Dev Kit)
3. Choose your device in the Connected Devices dropdown, and click on `Start`
4. Navigate to `Example Projects & Demos`. Then, in the context menu on the left, scroll down to `Capability`, and select `Machine Learning`.
5. Locate the demo you want to try out and click on `Run`. Note that the listed demos will depend on what board you've connected. The [demos.xml](demos.xml) file in this repo lists all available demos.

#### Simplicity Studio: Generating, building and flashing an application

Some applications in this repository include project templates which can be used to generate projects you can modify and try out. You can generate a new project based on project templates provided by this repository as follows:

1. Open the Simplicity Studio Launcher (the rocket button in the top right corner of the window)
2. Connect your device (e.g. an EFR32xG24 Dev Kit)
3. Choose your device in the Connected Devices dropdown, and click on `Start`
4. Navigate to `Example Projects & Demos`. Then, in the context menu on the left, scroll down to `Capability`, and select `Machine Learning`.
5. Locate the template you want to try out and click on `Create`. You can typically leave the settings to their defaults, and click on `Finish` to set up the project for your board. Note that the listed templates will depend on what board you've connected. The [templates.xml](templates.xml) file in this repo lists all available project templates.
6. Inside your project, you can then compile and run the code from the window context menu:
   - Click on `Project`→`Build Project` to compile the code
   - Click on `Run`→`Debug` and select your connected device to flash the compiled firmware onto your device

**You need to perform this step every time you make changes to the extension.**

Now you can reference the components in this extension, in projects that are generated using the above Gecko SDK.

## Testing

You can find scripts for testing the repository under `tests/`. These are CMake-based. The tests can be run through the provided build container or natively.

The scripts define two kinds of tests:

- Unit tests: Standard unit testing using GoogleTest
- Application builds: Verifies that the bundled applications compile when targeting specific Silabs development kits

### Testing: Using build container

After building the [#Build container](#build-container) image, you can configure, build and run the tests by running

```sh
docker run mla-builder tests/run.sh /opt/gecko_sdk
```

### Testing: Using command line tools

To run the tests natively using command line tools you need to install [CMake](https://cmake.org/). In addition, you'll need the prerequisites described in [#Command line tools: Installing the prerequites](#command-line-tools-installing-the-prerequisites).

#### Natively: Running the tests

You can configure, build and run the tests by running

```sh
# Configure build scripts
cmake -S tests -B tests/build
# Compile applications and test binaries
cmake --build tests/build
# Run tests
ctest --test-dir tests/build
```

## License

Certain files and directories have specific licensing terms which are clearly marked. Aside from that, content in this repository is generally available under the Zlib license. See [LICENSE](LICENSE) for more details.
