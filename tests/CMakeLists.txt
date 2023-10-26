
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

set(GTEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/gtest_main.cc)

macro(add_utest_kokkos TESTNAME TESTSRCS)
  set(testNameFinal ${TESTNAME})
  add_executable(${testNameFinal} ${TESTSRCS} ${GTEST_MAIN})
  target_link_libraries(${testNameFinal} ${PDA_ALL_TPLS_NEEDED} GTest::gtest_main)
  add_test(NAME ${testNameFinal} COMMAND ${testNameFinal})
endmacro()

set(NAMES "main")
foreach(id IN LISTS NAMES)
  set(TEST test_${id})
  add_utest_kokkos(${TEST} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST}.cc)
endforeach()
