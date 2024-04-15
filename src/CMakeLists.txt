cmake_minimum_required (VERSION 3.8)

add_executable(NearDupes.App "main.cpp")
set_target_properties(NearDupes.App
  PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
  )
target_link_libraries(NearDupes.App 
  PRIVATE
    murmurhash
    csv
    lmdbxx
  )
