project( shibie )
find_package( OpenCV REQUIRED )
add_executable( shibie shibie )
target_link_libraries( shibie ${OpenCV_LIBS} )

