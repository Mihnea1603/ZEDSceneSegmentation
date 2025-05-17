# ZED Scene Segmentation Setup Instructions

Set up and run the ZED 2 camera scene segmentation application on Windows using Visual Studio 2019/2022, CMake, CUDA Toolkit 12, and [ZED SDK 4.1.2](https://www.stereolabs.com/developers/release/4.1).

## Steps

1. **Configure CMake:**

    - Open CMake GUI.
    - Set source path to the downloaded repository root (where `CMakeLists.txt` is located).
    - Set build path to a `build` folder inside the root.
    - Configure, then generate for Visual Studio.

2. **Build and Run in Visual Studio:**

    - Open the `.sln` file in the `build` folder.
    - In Solution Explorer, right-click the `ZED_SVO_Playback` project and select "Set as Startup Project".
    - Build and run the solution in Release mode from Visual Studio.
