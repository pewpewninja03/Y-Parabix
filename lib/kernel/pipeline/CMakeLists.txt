#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: kernel.pipeline

if (ENABLE_PAPI)
    set(PAPI_CPP compiler/statistics/papi_instrumentation_logic.cpp)
endif()

parabix_add_module(
USES_Z3
NAME
    kernel.pipeline
SRC
    driver/driver.cpp
    driver/cpudriver.cpp
    # driver/NVPTXDriver.cpp
    compiler/analysis/add_analysis.cpp
    compiler/analysis/buffer_analysis.cpp
    compiler/analysis/consumer_analysis.cpp
    compiler/analysis/dataflow_analysis.cpp
    compiler/analysis/evolutionary_algorithm.cpp
    compiler/analysis/family_analysis.cpp
    compiler/analysis/internallygeneratedstreamset_analysis.cpp
    compiler/analysis/partitioning_analysis.cpp
    compiler/analysis/pipeline_graph_printers.cpp
    compiler/analysis/principal_input_analysis.cpp
    compiler/analysis/relationship_analysis.cpp
    compiler/analysis/scheduling_analysis.cpp
    compiler/analysis/simple_scheduling_analysis.cpp
    compiler/analysis/simple_variable_rate_analysis.cpp
    compiler/analysis/termination_analysis.cpp
    compiler/analysis/thread_local_analysis.cpp
    compiler/analysis/variable_rate_analysis.cpp
    compiler/analysis/zero_extend_analysis.cpp
    compiler/common/common.cpp
    compiler/codegen/buffer_management_logic.cpp
    compiler/codegen/buffer_manipulation_logic.cpp
    compiler/codegen/consumer_logic.cpp
    compiler/codegen/flow_control_logic.cpp
    compiler/codegen/kernel_execution_logic.cpp
    compiler/codegen/kernel_family_logic.cpp
    compiler/codegen/kernel_io_calculation_logic.cpp
    compiler/codegen/kernel_logic.cpp
    compiler/codegen/multithreading_model_logic.cpp
    compiler/codegen/kernel_segment_processing_logic.cpp
    compiler/codegen/optimization_branch_logic.cpp
    compiler/codegen/partition_processing_logic.cpp
    compiler/codegen/pipeline_logic.cpp
    compiler/codegen/pipeline_optimization_logic.cpp
    compiler/codegen/repeatingstreamset_logic.cpp
    compiler/codegen/scalar_logic.cpp
    compiler/codegen/streamdata_illustrator_logic.cpp
    compiler/codegen/synchronization_logic.cpp
    compiler/codegen/termination_logic.cpp
    compiler/codegen/thread_local_management_logic.cpp
    compiler/internal/popcount_kernel.cpp
    compiler/internal/regionselectionkernel.cpp
    compiler/statistics/cycle_counter_logic.cpp
    compiler/statistics/histogram_generation_logic.cpp
    compiler/statistics/multithreading_report_logic.cpp
    compiler/statistics/buffer_expansion_instrumentation_logic.cpp
    ${PAPI_CPP}
    compiler/statistics/track_blocking_io_logic.cpp
    optimizationbranch/optimizationbranch.cpp
    pipeline_builder.cpp
    pipeline_kernel.cpp
    program_builder.cpp
DEPS
    objcache
    toolchain
    kernel.illustrator
)
