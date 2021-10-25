cmake_minimum_required(VERSION 3.1)

project( pcv1 LANGUAGES CXX )

find_package( OpenCV REQUIRED )




add_library(code 
    Pcv1.cpp
    Pcv1.h
)

set_target_properties(code PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_link_libraries(code 
    PUBLIC
        opencv_core opencv_highgui opencv_imgproc
)




add_executable(main 
    main.cpp 
)

target_link_libraries(main 
    PRIVATE
        code
)



add_executable(test 
    test.cpp 
)

target_link_libraries(test 
    PRIVATE
        code
)
