# Vision Frontend

This is an SLC component that provides useful functions when working with images and deep learning. It currently covers the following main areas:

- Bounding boxes
- Histograms
- Images (data structure, cropping, resizing, etc.)

To use, assuming this extension is already installed, simply add the following into your `*.slcp` file:

```yaml
sdk_extension:
  - id: machine_learning_applications
    version: "1.0.0"
component:
    # ... other components
  - id: vision
    from: machine_learning_applications
```

After generating the project using SLC, you can include `sl_vision.h` in your source code:

```c
#include "sl_vision.h"

// ... your code
```

For more information on how to use it, please refer to the vision example applications.
