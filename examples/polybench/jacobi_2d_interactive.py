# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32, float32
import allo.ir.types as T


def jacobi_2d_np(A, B, TSTEPS):
    for t in range(TSTEPS):
        for i in range(1, A.shape[0] - 1):
            for j in range(1, A.shape[1] - 1):
                B[i, j] = 0.2 * (
                    A[i, j + 1] + A[i, j - 1] + A[i - 1, j] + A[i + 1, j] + A[i, j]
                )
        for i in range(1, A.shape[0] - 1):
            for j in range(1, A.shape[1] - 1):
                A[i, j] = 0.2 * (
                    B[i, j + 1] + B[i, j - 1] + B[i - 1, j] + B[i + 1, j] + B[i, j]
                )
    return A, B


def compute_A[T: (float32, int32), N: int32](A0: "T[N, N]", B0: "T[N, N]"):
    for i0, j0 in allo.grid(N - 2, N - 2, name="A"):
        B0[i0 + 1, j0 + 1] = 0.2 * (
            A0[i0, j0 + 1]
            + A0[i0 + 1, j0]
            + A0[i0 + 1, j0 + 1]
            + A0[i0 + 1, j0 + 2]
            + A0[i0 + 2, j0 + 1]
        )


def compute_B[T: (float32, int32), N: int32](B1: "T[N, N]", A1: "T[N, N]"):
    for i1, j1 in allo.grid(N - 2, N - 2, name="B"):
        A1[i1 + 1, j1 + 1] = 0.2 * (
            B1[i1, j1 + 1]
            + B1[i1 + 1, j1]
            + B1[i1 + 1, j1 + 1]
            + B1[i1 + 1, j1 + 2]
            + B1[i1 + 2, j1 + 1]
        )


def kernel_jacobi_2d[T: (float32, int32), N: int32](A: "T[N, N]", B: "T[N, N]"):
    for m in range(TSTEPS):
        compute_A(A, B)
        compute_B(B, A)


def build_baseline_model(concrete_type, TSTEPS, N):
    import os
    import allo
    from pathlib import Path
    
    prj_dir = "jacobi_2d_baseline.prj"
    prj_path = str(Path(__file__).parent / prj_dir)
    baseline_file = Path(__file__).parent / prj_dir / "out.prj" / "solution1" / "zero_cosim_model_impl.py"
    
    if baseline_file.exists():
        return baseline_file.read_text()
        
    print(f"\n[Baseline] Compiling uncustomized baseline CSynth to {prj_path}...")
    sch_base = allo.customize(kernel_jacobi_2d, instantiate=[concrete_type, N])
    # Mode is csynth to generate ADBs
    mod = sch_base.build(target="vitis_hls", mode="csyn", project=prj_path)
    mod()  # Actually run the HLS synthesis to generate the ADBs and artifacts
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        print("\n[Agent] Building initial zero_cosim_model from baseline artifacts...")
        sch_base.build_cosim_model(project=prj_path, api_key=api_key)
        
    return baseline_file.read_text() if baseline_file.exists() else None

def jacobi_2d(concrete_type, TSTEPS, N):
    import os
    import json
    
    baseline_code = build_baseline_model(concrete_type, TSTEPS, N)
    
    print("\n[Ground Truth] Baseline CSynth Cycles:")
    xml_file_base = os.path.join(os.path.dirname(__file__), "jacobi_2d_baseline.prj", "out.prj", "solution1", "syn", "report", "kernel_jacobi_2d_csynth.xml")
    if os.path.exists(xml_file_base):
        import xml.etree.ElementTree as ET
        lat = ET.parse(xml_file_base).getroot().find('.//PerformanceEstimates/SummaryOfOverallLatency/Average-caseLatency').text
        print(f"        -> Actual CSynth Latency (Cycles): {lat}")
        
    sch0 = allo.customize(compute_A, instantiate=[concrete_type, N])
    lb0 = sch0.reuse_at(sch0.A0, "i0")
    wb0 = sch0.reuse_at(lb0, "j0")
    sch0.pipeline("i0")
    sch0.partition(lb0, dim=0)
    sch0.partition(wb0, dim=0)

    if baseline_code and os.environ.get("OPENROUTER_API_KEY"):
        try:
            print("\n[Agent] Predicting cycle report for compute_A schedule...")
            rep = sch0.predict_performance(baseline_code).report_cycle()
            print("        ->", json.dumps(rep, indent=2))
        except Exception as e:
            print("        -> Prediction Failed:", e)
            
    print("\n[Ground Truth] Compiling compute_A schedule CSynth to jacobi_2d_compute_A.prj...")
    mod_hls0 = sch0.build(target="vitis_hls", mode="csyn", project="jacobi_2d_compute_A.prj")
    mod_hls0()
    xml_file0 = "jacobi_2d_compute_A.prj/out.prj/solution1/syn/report/compute_A_csynth.xml"
    if os.path.exists(xml_file0):
        import xml.etree.ElementTree as ET
        lat = ET.parse(xml_file0).getroot().find('.//PerformanceEstimates/SummaryOfOverallLatency/Average-caseLatency').text
        print(f"        -> Actual CSynth Latency (Cycles): {lat}")

    sch1 = allo.customize(compute_B, instantiate=[concrete_type, N])
    lb1 = sch1.reuse_at(sch1.B1, "i1")
    wb1 = sch1.reuse_at(lb1, "j1")
    sch1.pipeline("i1")
    sch1.partition(lb1, dim=0)
    sch1.partition(wb1, dim=0)

    if baseline_code and os.environ.get("OPENROUTER_API_KEY"):
        try:
            print("\n[Agent] Predicting cycle report for compute_B schedule...")
            rep = sch1.predict_performance(baseline_code).report_cycle()
            print("        ->", json.dumps(rep, indent=2))
        except Exception as e:
            print("        -> Prediction Failed:", e)
            
    print("\n[Ground Truth] Compiling compute_B schedule CSynth to jacobi_2d_compute_B.prj...")
    mod_hls1 = sch1.build(target="vitis_hls", mode="csyn", project="jacobi_2d_compute_B.prj")
    mod_hls1()
    xml_file1 = "jacobi_2d_compute_B.prj/out.prj/solution1/syn/report/compute_B_csynth.xml"
    if os.path.exists(xml_file1):
        import xml.etree.ElementTree as ET
        lat = ET.parse(xml_file1).getroot().find('.//PerformanceEstimates/SummaryOfOverallLatency/Average-caseLatency').text
        print(f"        -> Actual CSynth Latency (Cycles): {lat}")

    sch = allo.customize(kernel_jacobi_2d, instantiate=[concrete_type, N])
    sch.compose(sch0)
    sch.compose(sch1)
    sch.partition(sch.A, dim=2)
    sch.partition(sch.B, dim=2)
    
    if baseline_code and os.environ.get("OPENROUTER_API_KEY"):
        try:
            print("\n[Agent] Predicting cycle report for fully composed kernel...")
            rep = sch.predict_performance(baseline_code).report_cycle()
            print("        ->", json.dumps(rep, indent=2))
        except Exception as e:
            print("        -> Prediction Failed:", e)
            
    return sch


def test_jacobi_2d():
    # read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    # for CI test we use small problem size
    test_psize = "small"
    N = psize["jacobi_2d"][test_psize]["N"]
    TSTEPS = psize["jacobi_2d"][test_psize]["TSTEPS"]
    concrete_type = float32
    sch = jacobi_2d(concrete_type, TSTEPS, N)
    
    # functional correctness test
    print("\n[Correctness] Verifying LLVM JIT Model...")
    mod = sch.build()
    A = np.random.randint(10, size=(N, N)).astype(np.float32)
    B = np.random.randint(10, size=(N, N)).astype(np.float32)
    A_ref = A.copy()
    B_ref = B.copy()
    A_ref, B_ref = jacobi_2d_np(A_ref, B_ref, TSTEPS)
    mod(A, B)
    np.testing.assert_allclose(A, A_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(B, B_ref, rtol=1e-5, atol=1e-5)
    print("        -> Passed!")

    # ground truth csynth
    prj_dir = "jacobi_2d_optimized.prj"
    prj_path = os.path.join(os.path.dirname(__file__), prj_dir)
    print(f"\n[Ground Truth] Compiling fully customized schedule CSynth to {prj_path}...")
    mod_hls = sch.build(target="vitis_hls", mode="csyn", project=prj_path)
    mod_hls()  # Actually run the HLS synthesis
    
    # parse ground truth from out.prj/solution1/syn/report/kernel_jacobi_2d_csynth.xml
    import xml.etree.ElementTree as ET
    xml_file = os.path.join(prj_path, "out.prj", "solution1", "syn", "report", "kernel_jacobi_2d_csynth.xml")
    if os.path.exists(xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        estimated_clock = root.find('.//PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod').text
        min_cycles = root.find('.//PerformanceEstimates/SummaryOfOverallLatency/Average-caseLatency').text
        print(f"        -> Actual CSynth Latency (Cycles): {min_cycles}")
        print(f"        -> Actual CP: {estimated_clock} ns")


if __name__ == "__main__":
    test_jacobi_2d()
