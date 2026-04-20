# Silicon Labs Machine Learning Applications

Silicon Labs provides integrated hardware, software and development tools to help you quickly create secure, intelligent devices suitable for both industrial and commercial use cases. Our development platform has first class support for embedded machine learning (TinyML) model inference, backed by the [Tensorflow Lite for Microcontrollers (TFLM)](https://www.tensorflow.org/lite/microcontrollers) framework. We offer devices such as the [EFR32xG24](https://www.silabs.com/wireless/zigbee/efr32mg24-series-2-socs) dev kit that have hardware accelerators specifically built for [high-performant and energy efficient](https://mlcommons.org/en/inference-tiny-07/) AI/ML edge computing.

This repository contains a collection of embedded applications that leverage ML. You can use these to program your own Silicon Labs device, or as a starting point to develop your own TinyML application.

Feel free to open an issue if you have any questions or encounter problems, but take note that unless otherwise stated, all examples are considered to be EXPERIMENTAL QUALITY. The provided code has not been formally tested and is provided as-is. It is not suitable for production environments. In addition, there may be no bug maintenance planned for these resources. Silicon Labs may update the repository from time to time.

## Table of Contents  <!-- omit from toc -->

- [Silicon Labs Machine Learning Applications](#silicon-labs-machine-learning-applications)
  - [About](#about)
  - [Dependent SDKs](#dependent-sdks)
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

The repository is organized by use case category. All applications are self-contained and include their own documentation.

Within an application's directory you will generally find:

- Source code for training the ML model and exporting it to TensorFlow Lite or another trained model artifact
- Documentation on model training and usage
- Optional host-side tools or scripts for visualization, data collection, or evaluation, when applicable

## Dependent SDKs and Studio

This repository is validated with:

1. Simplicity Studio 6
2. Simplicity SDK (SiSDK) v2025.12.2
3. Silicon Labs AI/ML v2.2.1

## Building and running

There are multiple demo applications and project templates in this repository. A Dockerfile is provided for containerized builds and tests. For SiSDK 2025.12.2, project generation has been validated in Simplicity Studio. Other workflows should be revalidated on the target setup before being treated as equivalent.

### Build container

The Dockerfile at `build/Dockerfile` provides tooling for running containerized application builds and tests.

To build and run an application for your board using the build container, you will need to:

1. Install the prerequisites
2. Use the build container to generate a project for your board using Silicon Labs Configurator and compile it using Make
3. Copy the compiled application binaries to your host machine
4. Use Simplicity Commander on your host machine to flash the compiled application onto your device

#### Build container: Installing the prerequisites

For this repository, Simplicity Studio project generation has been validated for SiSDK 2025.12.2. If you plan to rely on the build container for project generation and firmware build, validate that workflow separately against the target SDK and tool versions first.

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
    DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t mla-builder -f build/Dockerfile .
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

A native command-line workflow can be used for supported applications, but the exact generation and build steps depend on the installed SDK packages, target board, and local toolchain configuration.

For this repository, Simplicity Studio project generation has been validated for SiSDK 2025.12.2. If you plan to use a CLI workflow as your primary path, validate the full generate, build, and flash sequence on your setup first.

#### Command line tools: Installing the prerequisites

1. To flash binaries onto your device, install:

   - `commander` ([Simplicity Commander](https://www.silabs.com/developers/mcu-programming-options#programming))

2. To generate and build projects natively, install:

   - `slt`
   - `slc`
   - `cmake`
   - Arm GNU Embedded Toolchain (`arm-none-eabi`)

3. After installing the tools, make sure they are available in your `PATH`.

4. Install or make available the SDK packages required by the repository:

   - Simplicity SDK (SiSDK) `2025.12.2`
   - Silicon Labs AI/ML `2.2.1`

5. Clone this repository locally:

   ```sh
   git clone https://github.com/SiliconLabsSoftware/machine_learning_applications.git
   ```

> Note: This repository does not need to be cloned under an SDK `extension/` directory when using workflows that accept explicit SDK or extension paths.

#### Command line tools: Generating, building and flashing an application

A native CLI workflow typically uses `slc` for project generation and CMake for firmware build. The exact commands depend on the installed SDK packages, selected target board, and local environment.

### Simplicity Studio

To build and run demos for your board using Simplicity Studio, you will need to:

1. Add this repository as an SDK extension in Simplicity Studio
2. Select a supported demo or project template for your target board
3. Generate, build, and flash the project from the Studio workspace

#### Simplicity Studio: Adding an external repository

Simplicity Studio supports adding Simplicity SDK extensions that provide project templates, prebuilt demos, and software components. To add this repository as an SDK extension:

1. Download the code, either by:
   - Cloning the repository with `git`:

     ```sh
     git clone https://github.com/SiliconLabsSoftware/machine_learning_applications.git
     ```

   - Or downloading the repository archive and extracting it locally

2. Open Simplicity Studio
3. Open the Settings panel in Simplicity Studio
4. Select `SDKs`
5. Select your Simplicity SDK installation, then click `Add Extension...`
6. Click `Browse`, select the root directory of the downloaded repository, then add and trust the `Machine Learning Applications` extension
7. Click `Apply and Close`

> Note: Studio or VS Code schema validation may still report stale `.slce` errors, such as a missing `sdk` property or rejecting `vendor`. Successful project generation is the authoritative check.

#### Simplicity Studio: Flashing prebuilt demos

Some applications in this repository include prebuilt demo binaries that can be flashed onto your device without creating a local project.

To flash a prebuilt demo:

1. Open the Simplicity Studio Launcher
2. Connect your device, for example an EFR32xG24 Dev Kit
3. Select the connected device and click `Start`
4. Open `Example Projects & Demos`
5. In the left-side filters, select `Capability` -> `Machine Learning`
6. Locate the demo you want to try and click `Run`

The available demos depend on the connected board. See [demos.xml](demos.xml) for the full list.

#### Simplicity Studio: Generating, building and flashing an application

Some applications in this repository provide project templates that can be generated and modified in Simplicity Studio.

To generate and use one of these projects:

1. Open the Simplicity Studio Launcher
2. Connect your device, for example an EFR32xG24 Dev Kit
3. Select the connected device and click `Start`
4. Open `Example Projects & Demos`
5. In the left-side filters, select `Capability` -> `Machine Learning`
6. Locate the template you want to use and click `Create`
7. Review the project settings, then click `Finish` to generate the project for your board
8. Build and flash the generated project from the Studio workspace

The templates shown depend on the connected board. See [templates.xml](templates.xml) for the full list of available project templates.

**Regenerate the project after making changes to extension metadata or extension-provided components.**

After the extension has been added, its components can be referenced by projects generated against the selected Simplicity SDK installation.

## Testing

You can find scripts for testing the repository under `tests/`. These are CMake-based. The tests can be run through the provided build container or natively.

The scripts define two kinds of tests:

- Unit tests: Standard unit testing using GoogleTest
- Application builds: Verifies that the bundled applications compile when targeting specific Silicon Labs development kits

### Testing: Using build container

After building the [Build container](#build-container) image, you can configure, build, and run the tests using the scripts under `tests/`.

> Note: Revalidate the container-based test workflow against the target SiSDK and tool versions before relying on it as the primary verification path.

### Testing: Using command line tools

To run the tests natively using command line tools, install [CMake](https://cmake.org/). In addition, install the prerequisites described in [Command line tools: Installing the prerequisites](#command-line-tools-installing-the-prerequisites).

#### Natively: Running the tests

You can configure, build, and run the tests by running

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
