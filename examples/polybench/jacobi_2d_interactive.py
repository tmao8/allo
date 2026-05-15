import allo
from allo.ir.types import int32, float32


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


TSTEPS = 40
N_SIZE = 90


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
    mod = sch_base.build(target="vitis_hls", mode="csyn", project=prj_path)
    mod()
    
    print("\n[Agent] Building initial zero_cosim_model from baseline artifacts...")
    sch_base.build_cosim_model(project=prj_path)
        
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

    if baseline_code:
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

    if baseline_code:
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
    
    if baseline_code:
        try:
            print("\n[Agent] Predicting cycle report for fully composed kernel...")
            rep = sch.predict_performance(baseline_code).report_cycle()
            print("        ->", json.dumps(rep, indent=2))
        except Exception as e:
            print("        -> Prediction Failed:", e)
            
    return sch


def test_jacobi_2d():
    import os
    import numpy as np
    
    concrete_type = float32
    N = N_SIZE

    sch = jacobi_2d(concrete_type, TSTEPS, N)

    mod = sch.build(target="vitis_hls", mode="csyn", 
                    project=os.path.join(os.path.dirname(__file__), "jacobi_2d_optimized.prj"))
    mod()

    xml_opt = os.path.join(os.path.dirname(__file__), "jacobi_2d_optimized.prj/out.prj/solution1/syn/report/kernel_jacobi_2d_csynth.xml")
    if os.path.exists(xml_opt):
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_opt)
        lat_node = tree.getroot().find('.//PerformanceEstimates/SummaryOfOverallLatency/Average-caseLatency')
        cp_node = tree.getroot().find('.//PerformanceEstimates/SummaryOfTimingInformation/EstimatedClockPeriod')
        if lat_node is not None:
            print(f"        -> Actual CSynth Latency (Cycles): {lat_node.text}")
        if cp_node is not None:
            print(f"        -> Actual CP: {cp_node.text} ns")

    # Verify correctness with LLVM backend
    print("\n[Correctness] Verifying LLVM JIT Model...")
    mod_llvm = sch.build()
    A = np.random.uniform(size=(N, N)).astype(np.float32)
    B = np.random.uniform(size=(N, N)).astype(np.float32)
    A_ref = A.copy()
    B_ref = B.copy()
    for t in range(TSTEPS):
        B_new = B_ref.copy()
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                B_new[i, j] = 0.2 * (A_ref[i, j] + A_ref[i, j - 1] + A_ref[i, j + 1] + A_ref[i + 1, j] + A_ref[i - 1, j])
        B_ref = B_new
        A_new = A_ref.copy()
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                A_new[i, j] = 0.2 * (B_ref[i, j] + B_ref[i, j - 1] + B_ref[i, j + 1] + B_ref[i + 1, j] + B_ref[i - 1, j])
        A_ref = A_new

    mod_llvm(A, B)
    np.testing.assert_allclose(A, A_ref, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(B, B_ref, rtol=1e-3, atol=1e-3)
    print("        -> Passed!")


if __name__ == "__main__":
    test_jacobi_2d()
