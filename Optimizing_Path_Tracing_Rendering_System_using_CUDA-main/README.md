# Optimizing_Path_Tracing_Rendering_System_using_CUDA

How to compile and run the code

To compile:

    $nvcc kernel_ee451.cu -o <ouput_filename>

To run the exe output file:

    $./<output_filename>


// Kernel_ee451.cu (for parallel execution)

There are 3 different scenes and 2 different camera placement for its corresponding scenes

To check the ouput for different scenes, go to the function create_world and comment the scenes you are not using and corresponding camera function in the same create_world function. We have commented inside the function for which scene which camera parameters are being used.


// Parameters to change for different ouputs:

    1. Thread x and y dimensions
    2. Resolution height and width
    3. Num of samples per pixel


// Kernel_serial.cu (for serial execution)

Compiling and running this code is similar to the above instructions. The only change in parameters we have to consider while experimenting:

    // Parameters to change for different ouputs:

        1. Resolution height and width
        2. Num of samples per pixel

//(Note: Scene 0 in code represents Scene 1 in report
//           Scene 1 in code represents Scene 2 in report
//           Scene 2 in code represents Scene 3 in report)

CONFIGURATION OF THE SYSTEM:

    1. NVIDIA GEFORCE RTX 3070 8GB
    2. 16 GB RAM
    3. AMD RYZEN 9 5900HX 8 CORE PROCESSOR
    4. OS- WINDOWS 11
    5. CUDA Version 11.5.119
    6. Microsoft Visual Studio v1.66.0
    7. Nvidia Nsight Compute v2022.1.1
